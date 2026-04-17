import argparse
from pathlib import Path

import cv2


AUTO_CODECS = ("avc1", "H264", "mp4v")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Blur and compress a video by reducing detail, resolution, and optional fps."
    )
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output video path. Default: <input_stem>_small.mp4 next to the input file.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Resize scale for width and height. Default: 0.5",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=9,
        help="Odd Gaussian blur kernel size. Use 1 to disable blur. Default: 9",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=0.0,
        help="Gaussian blur sigma. Default: 0.0 (OpenCV auto)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=15.0,
        help="Target output fps. Use 0 to keep original fps. Default: 15",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="auto",
        help="FourCC codec, e.g. auto, avc1, H264, mp4v. Default: auto",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=30,
        help="Print progress every N written frames. Default: 30",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    return parser.parse_args()


def ensure_even(value):
    value = max(2, int(round(value)))
    return value if value % 2 == 0 else value - 1


def normalize_kernel(value):
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1


def resolve_output_path(input_path, output_arg):
    if output_arg:
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_small.mp4")


def resolve_output_fps(source_fps, target_fps):
    if target_fps <= 0 or source_fps <= 0:
        return 1, source_fps if source_fps > 0 else 30.0
    if target_fps >= source_fps:
        return 1, source_fps
    frame_step = max(1, int(round(source_fps / target_fps)))
    return frame_step, source_fps / frame_step


def create_writer(output_path, width, height, fps, codec_name):
    candidates = [codec_name] if codec_name.lower() != "auto" else list(AUTO_CODECS)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for codec in candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()

    raise RuntimeError(f"Could not create VideoWriter for {output_path} with codecs: {candidates}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if args.scale <= 0:
        raise ValueError("--scale must be > 0")

    output_path = resolve_output_path(input_path, args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    blur_kernel = normalize_kernel(args.blur_kernel)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    target_width = ensure_even(source_width * args.scale)
    target_height = ensure_even(source_height * args.scale)
    frame_step, output_fps = resolve_output_fps(source_fps, args.target_fps)

    writer, used_codec = create_writer(output_path, target_width, target_height, output_fps, args.codec)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Source: {source_width}x{source_height} @ {source_fps:.2f} FPS, {total_frames} frames")
    print(f"Output: {target_width}x{target_height} @ {output_fps:.2f} FPS")
    print(f"Blur kernel: {blur_kernel}")
    print(f"Frame step: {frame_step}")
    print(f"Codec: {used_codec}")
    print("Note: audio is not preserved by this OpenCV-based re-encode.")

    read_frames = 0
    written_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_step > 1 and read_frames % frame_step != 0:
                read_frames += 1
                continue

            if blur_kernel > 1:
                frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), args.blur_sigma)

            if target_width != source_width or target_height != source_height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

            writer.write(frame)
            written_frames += 1
            read_frames += 1

            if written_frames % args.progress_every == 0:
                if total_frames > 0:
                    progress = read_frames / total_frames * 100
                    print(f"Written {written_frames} frames ({progress:.1f}% of source read)")
                else:
                    print(f"Written {written_frames} frames")
    finally:
        cap.release()
        writer.release()

    print(f"Done. Wrote {written_frames} frames to {output_path}")


if __name__ == "__main__":
    main()
