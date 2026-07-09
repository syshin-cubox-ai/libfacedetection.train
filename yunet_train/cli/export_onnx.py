import argparse
from pathlib import Path

import torch

from yunet_train.engine.onnx_export import export_model_to_onnx, parse_input_shape
from yunet_train.tasks.face import build_yunet, clean_inference_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default="yunet_n")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--shape", type=int, nargs="+", default=[640, 640])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    return parser.parse_args()


def _output_names() -> list[str]:
    names: list[str] = []
    for head in ("cls", "obj", "bbox", "kps"):
        names.extend([f"{head}_{stride}" for stride in (8, 16, 32)])
    return names


def _flatten_export_outputs(
    model: torch.nn.Module, image: torch.Tensor
) -> list[torch.Tensor]:
    cls_scores, bbox_preds, objectnesses, kps_preds = model(image)
    batch_size = image.shape[0]
    cls = [
        pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1).sigmoid()
        for pred in cls_scores
    ]
    obj = [
        pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1).sigmoid()
        for pred in objectnesses
    ]
    bbox = [pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for pred in bbox_preds]
    kps = [pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 10) for pred in kps_preds]
    return cls + obj + bbox + kps


def main():
    args = parse_args()
    output_path = export_model_to_onnx(
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        build_model=build_yunet,
        output_file=args.output_file or args.checkpoint.with_suffix(".onnx"),
        input_shape=parse_input_shape(args.shape, batch=args.batch),
        output_names=_output_names(),
        flatten_outputs=_flatten_export_outputs,
        dynamic=args.dynamic,
        opset=args.opset,
        device=args.device,
        verify=args.verify,
        half=args.half,
        clean_state_dict=clean_inference_state_dict,
    )
    print(f"Successfully exported: {output_path}")


if __name__ == "__main__":
    main()
