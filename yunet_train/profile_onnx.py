import argparse
from pathlib import Path

import onnx_tool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_tool.model_profile(args.onnx_file)


if __name__ == "__main__":
    main()
