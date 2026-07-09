from pathlib import Path
from typing import Callable

import numpy as np
import onnxslim
import torch

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
    opset_version: int,
    device: torch.device | str,
    verify: bool,
    clean_state_dict: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    | None = None,
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolved_variant = variant or checkpoint.get("config", {}).get("variant", "yunet_n")
    model = build_model(resolved_variant)
    if clean_state_dict is not None:
        state_dict = clean_state_dict(checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=True)
    else:
        load_checkpoint(checkpoint_path, model=model, map_location="cpu")
    model.to(device).eval()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    example_input = torch.randn(input_shape, dtype=torch.float32, device=device)
    dynamic_axes = _dynamic_axes(output_names) if dynamic else None

    with torch.no_grad():
        torch.onnx.export(
            model,
            (example_input,),
            output_file,
            input_names=["input"],
            output_names=output_names,
            opset_version=opset_version,
            dynamo=False,
            dynamic_axes=dynamic_axes,
        )

    onnxslim.slim(str(output_file), output_model=str(output_file))
    if verify:
        verify_onnx(model, example_input, output_file, flatten_outputs)
    return output_file


def verify_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_file: Path,
    flatten_outputs: Callable[[torch.nn.Module, torch.Tensor], list[torch.Tensor]],
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
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"ONNX output {idx} differs from PyTorch",
        )
    print("The numerical values are close between PyTorch and ONNX")


def parse_input_shape(shape: list[int]) -> tuple[int, int, int, int]:
    if len(shape) == 1:
        height = width = shape[0]
    elif len(shape) == 2:
        height, width = shape
    else:
        raise ValueError("--shape expects one int or two ints")
    return (1, 3, height, width)


def _dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    dynamic_axes = {name: {0: "batch", 1: "dim"} for name in output_names}
    dynamic_axes["input"] = {0: "batch", 2: "height", 3: "width"}
    return dynamic_axes
