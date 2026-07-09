import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime
import onnxslim
import torch
from onnxruntime.transformers import float16

from yunet_train.engine.checkpoint import load_checkpoint


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
    opset: int,
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
    dynamic_axes = _dynamic_axes(output_names) if dynamic else None

    # dry runs
    for _ in range(2):
        model(example_input)
    with torch.no_grad():
        output_shapes = tuple(
            tuple(output.shape) for output in flatten_outputs(model, example_input)
        )

    print(
        f"PyTorch: starting from '{checkpoint_path}' ({_file_size(checkpoint_path):.1f} MB)\n"
        f"{input_shape=} BCHW\n"
        f"{output_shapes=}"
    )

    t_onnx = time.time()
    print(f"\nONNX: starting export with onnx {onnx.__version__} opset {opset}...")
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

    try:
        print(f"ONNX: slimming with onnxslim {onnxslim.__version__}...")
        model_onnx = onnxslim.slim(model_onnx)
    except Exception as e:
        print(f"WARNING: ONNX: simplifier failure: {e}")

    for key, value in _metadata(model, resolved_variant, input_shape, dynamic).items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = key, str(value)

    if getattr(model_onnx, "ir_version", 0) > 10:
        print(
            f"ONNX: limiting IR version {model_onnx.ir_version} to 10 for ONNXRuntime compatibility..."
        )
        model_onnx.ir_version = 10

    if half:
        try:
            print("ONNX: converting to FP16...")
            model_onnx = float16.convert_float_to_float16(
                model_onnx, keep_io_types=True
            )
        except Exception as e:
            print(f"WARNING: ONNX: FP16 conversion failure: {e}")

    onnx.save(model_onnx, str(output_file))

    if verify:
        verify_input = torch.randn(input_shape, dtype=torch.float32, device=device)
        rtol, atol = (1e-2, 1e-2) if half else (1e-3, 1e-5)
        verify_onnx(
            model, verify_input, output_file, flatten_outputs, rtol=rtol, atol=atol
        )
        print("ONNX: the outputs are all close.")

    print(
        f"ONNX: export success ✅ {time.time() - t_onnx:.1f}s, "
        f"saved as '{output_file}' ({_file_size(output_file):.1f} MB)"
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
    with torch.no_grad():
        torch_outputs = [
            output.detach().cpu().numpy()
            for output in flatten_outputs(model, example_input)
        ]
    session = onnxruntime.InferenceSession(
        output_file, providers=["CPUExecutionProvider"]
    )
    onnx_outputs = session.run(
        None, {session.get_inputs()[0].name: example_input.detach().cpu().numpy()}
    )
    if len(torch_outputs) != len(onnx_outputs):
        raise AssertionError(
            f"ONNX output count mismatch: torch={len(torch_outputs)} onnx={len(onnx_outputs)}"
        )
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        np.testing.assert_allclose(torch_output, onnx_output, rtol, atol)


def _file_size(path: Path) -> float:
    """Return the size of a file in mebibytes (MiB)."""
    return path.stat().st_size / (1 << 20)


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
