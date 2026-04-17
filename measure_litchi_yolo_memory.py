import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_imgsz(values):
    if len(values) == 1:
        return int(values[0])
    if len(values) == 2:
        return [int(values[0]), int(values[1])]
    raise ValueError("--imgsz expects one value or two values, e.g. --imgsz 1024 or --imgsz 960 544")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure GPU memory usage of litchi-YOLO.")
    parser.add_argument("--model", type=str, default="models/yolov11-litchi.engine", help="Model path: .engine or .pt")
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Optional input video/image path. If omitted, a synthetic blank video is generated.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        default=["1024"],
        help="Inference image size. Use one value for square or two for height width.",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device, e.g. 0 or cuda:0")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--frames", type=int, default=120, help="Synthetic video frames if --source is omitted")
    parser.add_argument("--width", type=int, default=960, help="Synthetic video width")
    parser.add_argument("--height", type=int, default=544, help="Synthetic video height")
    parser.add_argument("--fps", type=int, default=25, help="Synthetic video fps")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames before measuring")
    parser.add_argument("--output", type=str, default="", help="Optional output json path")
    return parser.parse_args()


def parse_device_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return int(device)


def create_blank_video(width: int, height: int, frames: int, fps: int) -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix="litchi_mem_"))
    video_path = temp_dir / "blank.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.write(blank)
    writer.release()
    return str(video_path)


class GPUMemoryMonitor:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.pid = os.getpid()
        self.backend = None
        self.handle = None
        self.nvml = None
        self.peak_mb = 0.0

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.backend = "pynvml"
        except Exception:
            try:
                subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={device_index}",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.backend = "nvidia-smi"
            except Exception:
                self.backend = None

    def sample(self) -> float | None:
        value = None
        if self.backend == "pynvml":
            try:
                procs = self.nvml.nvmlDeviceGetComputeRunningProcesses_v3(self.handle)
            except Exception:
                try:
                    procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                except Exception:
                    procs = []
            used_bytes = 0
            for proc in procs:
                if proc.pid == self.pid:
                    used_bytes += int(getattr(proc, "usedGpuMemory", 0))
            value = used_bytes / (1024 ** 2)
        elif self.backend == "nvidia-smi":
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.device_index}",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                used_mb = 0.0
                for line in result.stdout.strip().splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == self.pid:
                        used_mb += float(parts[1])
                value = used_mb
            except Exception:
                value = None

        if value is not None:
            self.peak_mb = max(self.peak_mb, value)
        return value

    def close(self):
        if self.backend == "pynvml" and self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


def main():
    args = parse_args()
    args.imgsz = parse_imgsz(args.imgsz)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    source = args.source or create_blank_video(args.width, args.height, args.frames, args.fps)
    monitor = GPUMemoryMonitor(parse_device_index(args.device))
    if monitor.backend is None:
        raise RuntimeError("GPU memory monitor unavailable. Install pynvml or ensure nvidia-smi is available.")

    t0 = time.perf_counter()
    model = YOLO(str(model_path), task="detect")
    model_load_sec = time.perf_counter() - t0
    load_memory_mb = monitor.sample()

    predict_kwargs = {
        "source": source,
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "stream": True,
        "save": False,
        "show": False,
        "verbose": False,
    }

    processed_frames = 0
    warmup_frames = max(args.warmup, 0)
    infer_start = None
    inference_peak_mb = load_memory_mb or 0.0

    for result in model.predict(**predict_kwargs):
        processed_frames += 1
        current_mb = monitor.sample()
        if current_mb is not None:
            if processed_frames <= warmup_frames:
                inference_peak_mb = max(inference_peak_mb, current_mb)
            else:
                inference_peak_mb = max(inference_peak_mb, current_mb)
        if processed_frames == warmup_frames + 1 and infer_start is None:
            infer_start = time.perf_counter()
        _ = result.boxes

    total_elapsed = None
    measured_frames = max(processed_frames - warmup_frames, 0)
    if infer_start is not None and measured_frames > 0:
        total_elapsed = time.perf_counter() - infer_start

    fps = measured_frames / total_elapsed if total_elapsed and total_elapsed > 0 else None
    latency_ms = 1000.0 / fps if fps and fps > 0 else None

    summary = {
        "model": str(model_path),
        "source": source,
        "device": args.device,
        "imgsz": args.imgsz,
        "memory_backend": monitor.backend,
        "model_load_time_sec": round(model_load_sec, 4),
        "model_load_gpu_memory_mb": round(load_memory_mb, 2) if load_memory_mb is not None else None,
        "inference_gpu_peak_memory_mb": round(inference_peak_mb, 2) if inference_peak_mb is not None else None,
        "processed_frames": processed_frames,
        "warmup_frames": warmup_frames,
        "measured_frames": measured_frames,
        "inference_time_sec": round(total_elapsed, 4) if total_elapsed is not None else None,
        "inference_fps": round(fps, 2) if fps is not None else None,
        "avg_latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
    }

    monitor.close()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
