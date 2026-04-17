import argparse
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from region_count import count_objects_in_region, get_center_region


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark tracking and counting FPS.")
    parser.add_argument("--model", type=str, default="runs/train/model/weights/best.pt", help="YOLO model path")
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Existing video path. If omitted, a blank synthetic video will be generated automatically.",
    )
    parser.add_argument("--tracker", type=str, default="ultralytics/cfg/trackers/botsort.yaml", help="Tracker yaml")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    parser.add_argument("--width", type=int, default=1280, help="Synthetic video width")
    parser.add_argument("--height", type=int, default=720, help="Synthetic video height")
    parser.add_argument("--frames", type=int, default=300, help="Synthetic video frame count")
    parser.add_argument("--fps", type=int, default=30, help="Synthetic video fps")
    parser.add_argument("--device", type=str, default="", help="Device passed to YOLO, e.g. 0 or cpu")
    return parser.parse_args()


def create_blank_video(width: int, height: int, frames: int, fps: int) -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix="tracking_fps_"))
    video_path = temp_dir / "blank.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.write(blank_frame)
    writer.release()
    return str(video_path)


def benchmark(model: YOLO, video_path: str, tracker: str, imgsz: int, device: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    region = get_center_region(width, height)
    counted_objects = set()
    tracking_time = 0.0
    counting_time = 0.0
    processed_frames = 0
    detected_boxes_total = 0

    track_kwargs = {
        "source": video_path,
        "tracker": tracker,
        "imgsz": imgsz,
        "save": False,
        "show": False,
        "stream": True,
        "verbose": False,
    }
    if device:
        track_kwargs["device"] = device

    wall_start = time.perf_counter()
    results = model.track(**track_kwargs)
    results_iter = iter(results)

    while True:
        t0 = time.perf_counter()
        try:
            result = next(results_iter)
        except StopIteration:
            break
        tracking_time += time.perf_counter() - t0
        processed_frames += 1

        boxes = result.boxes

        t1 = time.perf_counter()
        detected_boxes = []
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detected_boxes.append((x1, y1, x2, y2, int(box.id[0].item())))
        counted_objects, new_count = count_objects_in_region(detected_boxes, region, counted_objects)
        counting_time += time.perf_counter() - t1

        detected_boxes_total += len(detected_boxes)

    wall_time = time.perf_counter() - wall_start

    effective_frames = processed_frames or total_frames
    tracking_fps = effective_frames / tracking_time if tracking_time > 0 else 0.0
    counting_fps = effective_frames / counting_time if counting_time > 0 else 0.0
    end_to_end_fps = effective_frames / wall_time if wall_time > 0 else 0.0

    return {
        "video_path": video_path,
        "resolution": f"{width}x{height}",
        "video_frames": total_frames,
        "processed_frames": processed_frames,
        "source_fps": round(source_fps, 2),
        "tracked_boxes_total": detected_boxes_total,
        "counted_objects_total": len(counted_objects),
        "new_counts_total": len(counted_objects),
        "tracking_time_sec": round(tracking_time, 4),
        "counting_time_sec": round(counting_time, 4),
        "wall_time_sec": round(wall_time, 4),
        "tracking_fps": round(tracking_fps, 2),
        "counting_fps": round(counting_fps, 2),
        "end_to_end_fps": round(end_to_end_fps, 2),
    }


def main():
    args = parse_args()
    video_path = args.video or create_blank_video(args.width, args.height, args.frames, args.fps)
    model = YOLO(args.model)

    metrics = benchmark(
        model=model,
        video_path=video_path,
        tracker=args.tracker,
        imgsz=args.imgsz,
        device=args.device,
    )

    print("=== Tracking Count FPS Benchmark ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
