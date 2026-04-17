import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from region_count import count_objects_in_region, get_center_region


def parse_args():
    parser = argparse.ArgumentParser(description="Edge deployment for litchi tracking and counting with TensorRT.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov11-litchi.engine",
        help="TensorRT engine path exported from litchi-YOLO.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: camera id, video path, or RTSP/HTTP stream.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="ultralytics/cfg/trackers/botsort.yaml",
        help="Tracker yaml path.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--device", type=str, default="0", help="Inference device, e.g. 0.")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold for tracking.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument("--save-video", action="store_true", help="Save the rendered tracking video.")
    parser.add_argument("--show", action="store_true", help="Display the rendered frames in a window.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/edge_deploy",
        help="Directory for output video and json summary.",
    )
    parser.add_argument(
        "--region-json",
        type=str,
        default="",
        help='Optional custom counting region json, e.g. {"x1":100,"y1":50,"x2":900,"y2":700}.',
    )
    parser.add_argument(
        "--skip-detector-mem",
        action="store_true",
        help="Skip detector-only GPU memory measurement pass.",
    )
    return parser.parse_args()


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def load_region(region_json: str, width: int, height: int):
    if not region_json:
        return get_center_region(width, height)

    region_data = json.loads(region_json)
    return [(region_data["x1"], region_data["y1"]), (region_data["x2"], region_data["y2"])]


def is_replayable_source(raw_source: str) -> bool:
    path = Path(raw_source)
    return path.exists() and path.is_file()


def parse_device_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return int(device)


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
        memory_mb = None
        if self.backend == "pynvml":
            try:
                processes = self.nvml.nvmlDeviceGetComputeRunningProcesses_v3(self.handle)
            except Exception:
                try:
                    processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                except Exception:
                    processes = []
            used_bytes = 0
            for process in processes:
                if process.pid == self.pid:
                    used_bytes += int(getattr(process, "usedGpuMemory", 0))
            memory_mb = used_bytes / (1024 ** 2)
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
                memory_mb = used_mb
            except Exception:
                memory_mb = None

        if memory_mb is not None:
            self.peak_mb = max(self.peak_mb, memory_mb)
        return memory_mb

    def close(self):
        if self.backend == "pynvml" and self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


def measure_detector_peak_memory(model_path: Path, source, args) -> tuple[float | None, str]:
    if args.skip_detector_mem:
        return None, "skipped by user"
    if not is_replayable_source(args.source):
        return None, "detector-only memory is only measured for replayable video files"

    monitor = None
    try:
        monitor = GPUMemoryMonitor(parse_device_index(args.device))
        if monitor.backend is None:
            return None, "GPU memory monitor unavailable (install pynvml or ensure nvidia-smi is available)"

        model = YOLO(str(model_path), task="detect")
        predict_kwargs = {
            "source": source,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "stream": True,
            "save": False,
            "show": False,
            "verbose": False,
        }
        for _ in model.predict(**predict_kwargs):
            monitor.sample()
        monitor.sample()
        return round(monitor.peak_mb, 2), f"measured via {monitor.backend}"
    finally:
        if monitor is not None:
            monitor.close()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Engine not found: {model_path}")

    source = parse_source(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_peak_memory_mb, detector_mem_note = measure_detector_peak_memory(model_path, source, args)

    model = YOLO(str(model_path), task="detect")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    region = load_region(args.region_json, width, height)
    counted_objects = set()
    total_frames = 0
    total_boxes = 0
    total_track_time = 0.0
    start_time = time.perf_counter()
    end_to_end_monitor = GPUMemoryMonitor(parse_device_index(args.device))

    writer = None
    if args.save_video:
        output_video = output_dir / "edge_tracking_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, source_fps or 30.0, (width, height))

    track_kwargs = {
        "source": source,
        "tracker": args.tracker,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "stream": True,
        "save": False,
        "show": False,
        "verbose": False,
    }

    for result in model.track(**track_kwargs):
        frame_start = time.perf_counter()
        total_frames += 1
        end_to_end_monitor.sample()

        frame = result.plot()
        boxes = result.boxes
        detected_boxes = []
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detected_boxes.append((x1, y1, x2, y2, int(box.id[0].item())))
        counted_objects, _ = count_objects_in_region(detected_boxes, region, counted_objects)
        total_boxes += len(detected_boxes)

        cv2.rectangle(frame, region[0], region[1], (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"Count: {len(counted_objects)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        elapsed = time.perf_counter() - start_time
        current_fps = total_frames / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            frame,
            f"FPS: {current_fps:.2f}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("Litchi Edge Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        total_track_time += time.perf_counter() - frame_start

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    wall_time = time.perf_counter() - start_time
    avg_fps = total_frames / wall_time if wall_time > 0 else 0.0
    avg_latency_ms = 1000.0 / avg_fps if avg_fps > 0 else 0.0
    end_to_end_monitor.sample()

    summary = {
        "model": str(model_path),
        "source": args.source,
        "tracker": args.tracker,
        "imgsz": args.imgsz,
        "device": args.device,
        "resolution": f"{width}x{height}",
        "frames": total_frames,
        "tracked_boxes_total": total_boxes,
        "counted_objects_total": len(counted_objects),
        "wall_time_sec": round(wall_time, 4),
        "pipeline_time_sec": round(total_track_time, 4),
        "end_to_end_fps": round(avg_fps, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "detector_gpu_peak_memory_mb": detector_peak_memory_mb,
        "detector_gpu_memory_note": detector_mem_note,
        "end_to_end_gpu_peak_memory_mb": round(end_to_end_monitor.peak_mb, 2) if end_to_end_monitor.backend else None,
        "end_to_end_gpu_memory_note": (
            f"measured via {end_to_end_monitor.backend}"
            if end_to_end_monitor.backend
            else "GPU memory monitor unavailable (install pynvml or ensure nvidia-smi is available)"
        ),
    }
    end_to_end_monitor.close()

    summary_path = output_dir / "edge_tracking_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Edge deployment finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
