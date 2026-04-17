import os
import numpy as np
from collections import defaultdict


def load_results(file_path, is_gt=False):
    """
    Load tracking or ground truth results from the txt file.
    The file format should be: [frame_idx, obj_id, x, y, w, h, conf]
    Ground truth file does not have `conf`, so we assume `conf` = 1 for GT.
    """
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            frame_idx = int(data[0])
            obj_id = int(data[1])
            x, y, w, h = data[2:6]
            if is_gt:
                conf = 1.0  # Default confidence for ground truth is 1
            else:
                conf = data[6]
            if frame_idx not in results:
                results[frame_idx] = []
            results[frame_idx].append((obj_id, x, y, w, h, conf))
    return results


def iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    box = (x, y, w, h)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(track_results, gt_results, iou_threshold=0.5):
    """
    Calculate MOTA, IDF1, ID Switches, and Track Fragmentation based on track results and ground truth.
    """
    FN = FP = ID_switches = TP = track_fragments = 0
    total_GT = sum(len(gt_results[frame]) for frame in gt_results)
    tracked_ids = {}  # 存储每个目标的最后一次出现的帧编号
    active_tracks = {}  # 用于记录当前正在跟踪的目标 ID

    id_switches_per_frame = defaultdict(int)

    for frame_idx in sorted(gt_results.keys()):
        gt_boxes = gt_results.get(frame_idx, [])
        track_boxes = track_results.get(frame_idx, [])

        matched_gt = set()
        matched_track = set()

        # 计算 TP 和 FP
        for track in track_boxes:
            best_iou = 0
            best_gt_idx = -1
            for i, gt in enumerate(gt_boxes):
                iou_score = iou(track[1:5], gt[1:5])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = i
            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_track.add(track[0])
                TP += 1
            else:
                FP += 1

        # 计算 FN（未匹配的 GT）
        FN += len(gt_boxes) - len(matched_gt)

        # 计算 ID 切换（ID Switch）
        for track_id in matched_track:
            if track_id in tracked_ids:
                last_frame = tracked_ids[track_id]
                if last_frame != frame_idx - 1:
                    ID_switches += 1
                    id_switches_per_frame[frame_idx] += 1
            tracked_ids[track_id] = frame_idx  # 更新最新的帧编号

        # 计算轨迹碎片（Track Fragmentation）
        for gt in gt_boxes:
            gt_id = gt[0]  # GT 目标 ID
            if gt_id in active_tracks:
                last_frame = active_tracks[gt_id]
                if frame_idx - last_frame > 1:  # 说明目标丢失了一些帧
                    track_fragments += 1
            active_tracks[gt_id] = frame_idx  # 更新目标的最新出现帧

    # MOTA 计算
    MOTA = 1 - (FN + FP + ID_switches) / total_GT

    # IDF1 计算
    IDF1 = 2 * TP / (2 * TP + FP + FN)

    return MOTA, IDF1, track_fragments, ID_switches

def calculate_hota(track_results, gt_results, iou_threshold=0.5):
    """
    Calculate HOTA (Higher Order Tracking Accuracy) based on track results and ground truth.
    HOTA = (MOTA + IDF1) / 2
    """
    # First calculate MOTA and IDF1
    MOTA, IDF1, _, _ = calculate_metrics(track_results, gt_results, iou_threshold)

    # Calculate HOTA as the average of MOTA and IDF1
    HOTA = (MOTA + IDF1) / 2

    return HOTA


def evaluate_folders(pred_folder, gt_folder):
    """
    Evaluate all tracking results in the prediction folder against ground truth in gt folder.
    It assumes that both folders contain txt files with matching names.
    """
    # Get all prediction files and ground truth files
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.txt') and f != "summary.txt"]  # 过滤掉summary.txt文件
    gt_files = sorted(f for f in os.listdir(gt_folder) if f.endswith('.txt'))

    # Print files for debugging
    print(f"Prediction files: {pred_files}")
    print(f"Ground truth files: {gt_files}")

    assert len(pred_files) == len(gt_files), "Prediction and GT files count do not match."

    # Initialize variables to accumulate overall MOTA, IDF1, HOTA, ID Switches, Track Fragmentation
    total_MOTA = total_IDF1 = total_HOTA = total_IDSW = total_track_fragments = 0
    num_files = len(pred_files)

    # Evaluate each file pair
    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, gt_file)

        print(f"Evaluating: {pred_file} vs {gt_file}")

        # Load tracking results and ground truth
        track_results = load_results(pred_path)
        gt_results = load_results(gt_path, is_gt=True)

        # Calculate MOTA, IDF1, Track Fragmentation, ID Switches, and HOTA for the current file
        MOTA, IDF1, track_fragments, ID_switches = calculate_metrics(track_results, gt_results)
        HOTA = calculate_hota(track_results, gt_results)

        # Accumulate metrics
        total_MOTA += MOTA
        total_IDF1 += IDF1
        total_HOTA += HOTA
        total_IDSW += ID_switches
        total_track_fragments += track_fragments

        # Print frame-level ID switches and track fragments for debugging
        print(f"Track Fragmentation: {track_fragments}")
        print(f"ID Switches: {ID_switches}")

    # Calculate average metrics across all files
    avg_MOTA = total_MOTA / num_files
    avg_IDF1 = total_IDF1 / num_files
    avg_HOTA = total_HOTA / num_files
    avg_IDSW = total_IDSW / num_files
    avg_track_fragments = total_track_fragments / num_files

    print("\nAverage Metrics Across All Files:")
    print(f"MOTA: {avg_MOTA:.4f}")
    print(f"IDF1: {avg_IDF1:.4f}")
    print(f"HOTA: {avg_HOTA:.4f}")
    print(f"Average ID Switches (IDSW): {avg_IDSW:.4f}")
    print(f"Average Track Fragmentation: {avg_track_fragments:.4f}")


if __name__ == "__main__":
    # Paths to the prediction and ground truth folders
    pred_folder = r"runs/litchi/region_count_dev_3"
    gt_folder = r"D:\Desktop\GT"

    # Evaluate the folders
    evaluate_folders(pred_folder, gt_folder)
