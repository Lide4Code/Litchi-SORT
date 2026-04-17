import warnings
import argparse
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import csv
import json
import subprocess
import av
from region_count_dev import count_objects_in_region, get_center_region

warnings.filterwarnings('ignore')

#  python track_mul.py --input videos/test.mp4,videos/test2.mp4 --output tracking_results --folder exp1

def parse_arguments():
    parser = argparse.ArgumentParser(description='Single Video Tracking & Counting')
    parser.add_argument('--input', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--folder', type=str, required=True, help='Experiment folder name')
    return parser.parse_args()

def convert_to_browser_friendly(input_path, output_path):
    try:
        input_container = av.open(input_path)
        output_container = av.open(output_path, mode='w')

        stream = input_container.streams.video[0]
        output_stream = output_container.add_stream('h264', rate=stream.average_rate)
        output_stream.width = stream.width
        output_stream.height = stream.height
        output_stream.pix_fmt = 'yuv420p'

        for frame in input_container.decode(video=0):
            output_container.mux(output_stream.encode(frame))

        output_container.close()
        input_container.close()
        os.remove(input_path)
        print(f"Re-encoded video saved to {output_path}")
    except Exception as e:
        print(f"PyAV failed: {e}")

def process_video(video_path, model, output_folder, save_video=False, show_video=False):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None, None

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    region = get_center_region(video_width, video_height)

    results = model.track(
        source=video_path,
        tracker='ultralytics/cfg/trackers/botsort.yaml',
        imgsz=1024,
        save=False,
        save_txt=False,
        show=False
    )

    output_txt_path = os.path.join(output_folder, f'{video_name}.txt')
    all_results = []
    counted_objects = set()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = os.path.join(output_folder, f"{video_name}_temp.mp4")
    out_video = cv2.VideoWriter(temp_video_path, fourcc, 30, (video_width, video_height)) if save_video else None

    progress_path = os.path.join(output_folder, f"{video_name}_progress.json")

    for frame_idx, result in enumerate(results):
        boxes = result.boxes
        if boxes is None:
            continue

        frame = result.plot()
        cv2.rectangle(frame, region[0], region[1], (0, 0, 255), 2)
        cv2.putText(frame, f'Objects in region: {len(counted_objects)}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for box in boxes:
            if box.xyxy.ndimension() == 2 and box.xyxy.shape[1] == 4:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                obj_id = int(box.id) if box.id is not None else -1
                detected_boxes = [(x1, y1, x2, y2, obj_id)]
                counted_objects, _ = count_objects_in_region(detected_boxes, region, counted_objects)

                all_results.append([frame_idx, obj_id, x1, y1, x2 - x1, y2 - y1, box.conf.item()])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {obj_id}' if obj_id != -1 else 'ID: Unknown',
                            (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save_video and out_video is not None:
            out_video.write(frame)
        if show_video:
            cv2.imshow('Detection with Region', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        with open(progress_path, 'w') as pf:
            json.dump({'frame': frame_idx + 1, 'total': total_frames}, pf)

    if all_results:
        with open(output_txt_path, 'w') as f:
            for line in all_results:
                f.write(' '.join(map(str, line)) + '\n')
        print(f"Saved results to {output_txt_path}")
    else:
        print(f"No valid detections in {video_name}")

    if save_video and out_video is not None:
        out_video.release()

    cap.release()
    if show_video:
        cv2.destroyAllWindows()

    if save_video and os.path.exists(temp_video_path):
        final_output_path = os.path.join(output_folder, f"{video_name}.mp4")
        convert_to_browser_friendly(temp_video_path, final_output_path)

    return video_name, len(counted_objects), total_frames, round(len(counted_objects) / total_frames, 2)

if __name__ == '__main__':
    args = parse_arguments()
    model = YOLO('models/yolov11-litchi.pt')

    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_file():
        print("Error: input must be a valid video file.")
        exit()

    output_folder = os.path.join(args.output, args.folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_name, count, total_frames, avg_count = process_video(str(input_path), model, output_folder, save_video=True, show_video=False)

    summary_data = [[video_name, count, total_frames, avg_count]] if video_name else []

    csv_output_path = os.path.join(output_folder, 'summary.csv')
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Video Name', 'Counted Objects in Region', 'Total Frames', 'Avg Count per Frame'])
        writer.writerows(summary_data)

    json_output_path = os.path.join(output_folder, 'tracking_summary.json')
    with open(json_output_path, 'w') as jsonfile:
        json.dump([
            {"video_name": video_name, "count": count, "frames": total_frames, "avg": avg_count}
        ], jsonfile, indent=2)

    with open(os.path.join(output_folder, 'exp_folder_name.txt'), 'w') as f:
        f.write(os.path.basename(output_folder))
