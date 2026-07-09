import argparse
from pathlib import Path

import torch

from yunet_train.engine.onnx_export import export_model_to_onnx, parse_input_shape
from yunet_train.tasks.pose import build_yunet_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default="yunet_n")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--shape", type=int, nargs="+", default=[640, 640])
    parser.add_argument("--kpt-shape", type=int, nargs=2, default=[17, 3])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    return parser.parse_args()


def _output_names() -> list[str]:
    names: list[str] = []
    for head in ("cls", "obj", "bbox", "kpt"):
        names.extend([f"{head}_{stride}" for stride in (8, 16, 32)])
    return names


def _flatten_export_outputs(
    model: torch.nn.Module,
    image: torch.Tensor,
    kpt_shape: tuple[int, int],
) -> list[torch.Tensor]:
    cls_scores, bbox_preds, objectnesses, kpt_preds = model(image)
    batch_size = image.shape[0]
    kpt_channels = kpt_shape[0] * kpt_shape[1]
    cls = [
        pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1).sigmoid()
        for pred in cls_scores
    ]
    obj = [
        pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1).sigmoid()
        for pred in objectnesses
    ]
    bbox = [pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for pred in bbox_preds]
    kpt = [
        pred.permute(0, 2, 3, 1).reshape(batch_size, -1, kpt_channels)
        for pred in kpt_preds
    ]
    return cls + obj + bbox + kpt


def main() -> None:
    args = parse_args()
    kpt_shape = tuple(args.kpt_shape)
    export_model_to_onnx(
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        build_model=lambda variant: build_yunet_pose(variant, kpt_shape=kpt_shape),
        output_file=args.output_file or args.checkpoint.with_suffix(".onnx"),
        input_shape=parse_input_shape(args.shape, batch=args.batch),
        output_names=_output_names(),
        flatten_outputs=lambda model, image: _flatten_export_outputs(
            model, image, kpt_shape
        ),
        dynamic=args.dynamic,
        opset=args.opset,
        device=args.device,
        verify=args.verify,
        half=args.half,
    )


if __name__ == "__main__":
    main()
