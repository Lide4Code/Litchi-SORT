import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export litchi-YOLO weights to TensorRT engine.")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolov11-litchi.pt",
        help="Path to the trained PyTorch weights.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Export image size.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id, e.g. 0.")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size.")
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace size in GiB.")
    parser.add_argument("--half", action="store_true", help="Export FP16 engine when supported.")
    parser.add_argument("--int8", action="store_true", help="Export INT8 engine when supported.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shape.")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Dataset yaml for INT8 calibration. Required when --int8 is used in practice.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if args.int8 and not args.data:
        raise ValueError("--data is required for INT8 export calibration.")

    model = YOLO(str(weights), task="detect")
    export_kwargs = {
        "format": "engine",
        "imgsz": args.imgsz,
        "device": args.device,
        "batch": args.batch,
        "workspace": args.workspace,
        "dynamic": args.dynamic,
        "half": args.half,
        "int8": args.int8,
    }
    if args.data:
        export_kwargs["data"] = args.data

    engine_path = model.export(**export_kwargs)

    print("TensorRT export finished.")
    print(f"weights: {weights}")
    print(f"engine: {engine_path}")
    print("")
    print("Recommendation:")
    print("Export the engine on the target edge device, or on a platform with the same GPU architecture,")
    print("TensorRT version, CUDA version, and driver stack for best compatibility.")


if __name__ == "__main__":
    main()
