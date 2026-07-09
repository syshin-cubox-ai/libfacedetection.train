from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxslim
import torch
from onnxruntime.transformers import float16

from yunet_train.engine.checkpoint import load_checkpoint


def best_onnx_opset(cuda: bool = False) -> int:
    """Return max ONNX opset for this torch version with ONNX fallback."""
    opset = torch.onnx.utils._constants.ONNX_MAX_OPSET - 1  # second-latest for safety
    opset = min(opset, 20)  # legacy TorchScript exporter caps at opset 20 in torch 2.9+
    if cuda:
        opset -= 2  # fix CUDA ONNXRuntime NMS squeeze op errors
    return min(opset, onnx.defs.onnx_opset_version())


def export_model_to_onnx(
    *,
    checkpoint_path: Path,
    variant: str | None,
    build_model: Callable[[str], torch.nn.Module],
    output_file: Path,
    input_shape: tuple[int, int, int, int],
    output_names: list[str],
    flatten_outputs: Callable[[torch.nn.Module, torch.Tensor], list[torch.Tensor]],
    dynamic: bool,
    opset: int | None,
    device: torch.device | str,
    verify: bool,
    half: bool = False,
    clean_state_dict: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    | None = None,
) -> Path:
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolved_variant = variant or checkpoint.get("config", {}).get("variant", "yunet_n")
    model = build_model(resolved_variant)
    if clean_state_dict is not None:
        state_dict = clean_state_dict(checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=True)
    else:
        load_checkpoint(checkpoint_path, model=model, map_location="cpu")
    model = model.to(device).float().eval()
    for p in model.parameters():
        p.requires_grad = False

    output_file.parent.mkdir(parents=True, exist_ok=True)
    example_input = torch.zeros(input_shape, dtype=torch.float32, device=device)
    for _ in range(2):  # dry runs
        model(example_input)

    opset = opset or best_onnx_opset(cuda=device.type == "cuda")
    dynamic_axes = _dynamic_axes(output_names) if dynamic else None

    torch.onnx.export(
        model,
        (example_input,),
        output_file,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    model_onnx = onnx.load(str(output_file))
    model_onnx = onnxslim.slim(model_onnx)

    for key, value in _metadata(model, resolved_variant, input_shape, dynamic).items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = key, str(value)

    if getattr(model_onnx, "ir_version", 0) > 10:
        # limit IR version to 10 for ONNXRuntime compatibility
        model_onnx.ir_version = 10

    if half:
        try:
            model_onnx = float16.convert_float_to_float16(
                model_onnx, keep_io_types=True
            )
        except Exception as e:
            print(f"WARNING: FP16 conversion failure: {e}")

    onnx.save(model_onnx, str(output_file))

    if verify:
        verify_input = torch.randn(input_shape, dtype=torch.float32, device=device)
        rtol, atol = (1e-2, 1e-2) if half else (1e-3, 1e-5)
        verify_onnx(
            model, verify_input, output_file, flatten_outputs, rtol=rtol, atol=atol
        )
    return output_file


def verify_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_file: Path,
    flatten_outputs: Callable[[torch.nn.Module, torch.Tensor], list[torch.Tensor]],
    rtol: float,
    atol: float,
) -> None:
    import onnxruntime

    with torch.no_grad():
        torch_outputs = [
            output.detach().cpu().numpy()
            for output in flatten_outputs(model, example_input)
        ]
    session = onnxruntime.InferenceSession(
        str(output_file), providers=["CPUExecutionProvider"]
    )
    onnx_outputs = session.run(
        None, {session.get_inputs()[0].name: example_input.detach().cpu().numpy()}
    )
    if len(torch_outputs) != len(onnx_outputs):
        raise AssertionError(
            f"ONNX output count mismatch: torch={len(torch_outputs)} onnx={len(onnx_outputs)}"
        )
    for idx, (torch_output, onnx_output) in enumerate(zip(torch_outputs, onnx_outputs)):
        np.testing.assert_allclose(
            onnx_output,
            torch_output,
            rtol=rtol,
            atol=atol,
        )
    print("The numerical values are close between PyTorch and ONNX")


def parse_input_shape(shape: list[int], batch: int = 1) -> tuple[int, int, int, int]:
    if len(shape) == 1:
        height = width = shape[0]
    elif len(shape) == 2:
        height, width = shape
    else:
        raise ValueError("--shape expects one int or two ints")
    return (batch, 3, height, width)


def _metadata(
    model: torch.nn.Module,
    variant: str,
    input_shape: tuple[int, int, int, int],
    dynamic: bool,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "description": f"YuNet {variant} model",
        "date": datetime.now().isoformat(),
        "batch": input_shape[0],
        "imgsz": list(input_shape[2:]),
        "dynamic": dynamic,
    }
    config = getattr(model, "config", None)
    strides = getattr(config, "strides", None)
    if strides:
        metadata["stride"] = int(max(strides))
    return metadata


def _dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    dynamic_axes = {name: {0: "batch", 1: "dim"} for name in output_names}
    dynamic_axes["input"] = {0: "batch", 2: "height", 3: "width"}
    return dynamic_axes
