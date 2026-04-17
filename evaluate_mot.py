import os
import numpy as np
from collections import defaultdict


def load_results(file_path, is_gt=False):
    """加载跟踪结果或GT结果"""
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            frame_idx = int(data[0])
            obj_id = int(data[1])
            x, y, w, h = data[2:6]
            conf = 1.0 if is_gt else data[6]

            if frame_idx not in results:
                results[frame_idx] = []
            results[frame_idx].append((obj_id, x, y, w, h, conf))
    return results


def iou(box1, box2):
    """计算两个框的 IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, y1_min, x1_max, y1_max = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min, x2_max, y2_max = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

    inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1, area2 = w1 * h1, w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(track_results, gt_results, iou_threshold=0.5):
    """计算 MOTA、IDF1、IDSwitch、Track Fragmentation"""
    FN = FP = ID_switches = TP = track_fragments = 0
    total_GT = sum(len(gt_results[frame]) for frame in gt_results)
    tracked_ids = {}  # 存储目标上次出现的帧
    active_tracks = {}  # 记录当前正在跟踪的目标 ID

    id_switches_per_frame = defaultdict(int)

    for frame_idx in sorted(gt_results.keys()):
        gt_boxes = gt_results.get(frame_idx, [])
        track_boxes = track_results.get(frame_idx, [])

        matched_gt, matched_track = set(), set()

        # 计算 TP 和 FP
        for track in track_boxes:
            best_iou, best_gt_idx = 0, -1
            for i, gt in enumerate(gt_boxes):
                iou_score = iou(track[1:5], gt[1:5])
                if iou_score > best_iou:
                    best_iou, best_gt_idx = iou_score, i
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
                if last_frame != frame_idx - 1:  # ID 不连续
                    ID_switches += 1
                    id_switches_per_frame[frame_idx] += 1
            tracked_ids[track_id] = frame_idx  # 更新最新的帧编号

        # 计算轨迹碎片（Track Fragmentation）
        for gt in gt_boxes:
            gt_id = gt[0]  # GT 目标 ID
            if gt_id in active_tracks:
                last_frame = active_tracks[gt_id]
                if frame_idx - last_frame > 1:  # 目标丢失了一些帧
                    track_fragments += 1
            active_tracks[gt_id] = frame_idx  # 更新目标的最新出现帧

    # 计算 MOTA
    MOTA = 1 - (FN + FP + ID_switches) / total_GT

    # 计算 IDF1
    IDF1 = 2 * TP / (2 * TP + FP + FN)

    return MOTA, IDF1, ID_switches, track_fragments


def calculate_hota(track_results, gt_results, iou_threshold=0.5):
    """计算 HOTA (Higher Order Tracking Accuracy)"""
    MOTA, IDF1, _, _ = calculate_metrics(track_results, gt_results, iou_threshold)
    return (MOTA + IDF1) / 2


def evaluate_single_file(pred_path, gt_path):
    """评估单个文件的指标"""
    print(f"Evaluating: {os.path.basename(pred_path)} vs {os.path.basename(gt_path)}")

    # 读取预测结果和GT
    track_results = load_results(pred_path)
    gt_results = load_results(gt_path, is_gt=True)

    # 计算各项指标
    MOTA, IDF1, ID_switches, track_fragments = calculate_metrics(track_results, gt_results)
    HOTA = calculate_hota(track_results, gt_results)

    print("\nEvaluation Results:")
    print(f"MOTA: {MOTA:.4f}")
    print(f"IDF1: {IDF1:.4f}")
    print(f"HOTA: {HOTA:.4f}")
    print(f"ID Switches (IDSW): {ID_switches}")
    print(f"Track Fragmentation: {track_fragments}")


if __name__ == "__main__":
    # 设置评估的文件路径
    pred_file = r"runs/test/test_14/litchi1.txt"
    gt_file = r"D:\Desktop\GT\litchi1.txt"

    evaluate_single_file(pred_file, gt_file)
