import warnings
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import argparse
from region_count_dev import count_objects_in_region, get_center_region

warnings.filterwarnings('ignore')
AUTO_CODECS = ("avc1", "H264", "mp4v")
MODEL_CANDIDATES = (
    'runs/train/model/weights/best.pt',
    'models/yolov11-litchi.pt',
)

def create_output_folder(base_path):
    """
    创建唯一的输出文件夹（如果基础路径已存在则自动递增）
    :param base_path: 用户指定的基础路径（如：runs/track/exp）
    :return: 唯一的新建文件夹路径
    """
    output_folder = base_path
    counter = 1
    while os.path.exists(output_folder):
        output_folder = f"{base_path}_{counter}"
        counter += 1
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8 目标跟踪与区域计数')
    parser.add_argument('--input', type=str, default='video/litchi1.mp4',
                        help='输入视频路径（默认：video/litchi1.mp4）')
    parser.add_argument('--output', type=str, default='runs/track/exp',
                        help='输出目录基础路径（默认：runs/track/exp）')
    parser.add_argument('--save-scale', type=float, default=1.0,
                        help='输出视频缩放比例，默认 1.0')
    parser.add_argument('--save-fps', type=float, default=0.0,
                        help='输出视频 FPS，默认 0 表示跟随原视频')
    parser.add_argument('--blur-kernel', type=int, default=1,
                        help='输出视频高斯模糊核大小，1 表示不模糊')
    parser.add_argument('--codec', type=str, default='auto',
                        help='输出视频编码器，如 auto、avc1、H264、mp4v')
    return parser.parse_args()


def ensure_even(value):
    value = max(2, int(round(value)))
    return value if value % 2 == 0 else value - 1


def normalize_kernel(value):
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1


def resolve_output_fps(source_fps, target_fps):
    source_fps = source_fps if source_fps > 0 else 30.0
    if target_fps <= 0 or target_fps >= source_fps:
        return 1, source_fps
    frame_step = max(1, int(round(source_fps / target_fps)))
    return frame_step, source_fps / frame_step


def create_video_writer(output_path, width, height, fps, codec_name):
    candidates = [codec_name] if codec_name.lower() != 'auto' else list(AUTO_CODECS)

    for codec in candidates:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()

    raise RuntimeError(f'无法创建视频写入器：{output_path}，尝试编码器 {candidates} 均失败')


def resolve_model_path():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'未找到可用模型，请确保以下任一路径存在：{MODEL_CANDIDATES}'
    )

# python3 track.py --input video/litchi2.mp4 --output runs/track/exp

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 初始化模型
    model_path = resolve_model_path()
    print(f"使用模型：{model_path}")
    model = YOLO(model_path)

    # 打开视频文件
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {args.input}")
        exit()

    # 获取视频信息
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"视频分辨率：{video_width}x{video_height}")
    print(f"原始 FPS：{source_fps:.2f}")

    # 生成唯一输出目录
    output_folder = create_output_folder(args.output)
    print(f"输出目录已创建：{output_folder}")

    # 初始化视频写入器
    video_name = Path(args.input).stem  # 从输入路径提取文件名（无扩展名）
    output_video_path = os.path.join(output_folder, f"{video_name}_result.mp4")
    save_width = ensure_even(video_width * args.save_scale)
    save_height = ensure_even(video_height * args.save_scale)
    blur_kernel = normalize_kernel(args.blur_kernel)
    frame_step, output_fps = resolve_output_fps(source_fps, args.save_fps)
    out_video, used_codec = create_video_writer(output_video_path, save_width, save_height, output_fps, args.codec)
    print(f"输出视频：{save_width}x{save_height} @ {output_fps:.2f} FPS，编码器：{used_codec}")

    # 执行目标跟踪（禁用自动保存）
    results = model.track(
        source=args.input,
        tracker='ultralytics/cfg/trackers/botsort.yaml',
        imgsz=1024,
        save=False,
        show=False,
        stream=True
    )

    # 初始化计数相关变量
    counted_objects = set()
    output_txt_path = os.path.join(output_folder, f'{video_name}.txt')
    all_results = []
    region = get_center_region(video_width, video_height)

    # 处理每一帧
    for frame_idx, result in enumerate(results):
        frame = result.plot()  # 获取带检测框的帧
        boxes = result.boxes

        # 绘制中心检测区域
        cv2.rectangle(frame, region[0], region[1], (0, 0, 255), 2)

        if boxes is not None:
            # 更新区域计数
            detected_boxes = [
                (box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3], int(box.id))
                for box in boxes if box.id is not None
            ]
            counted_objects, count = count_objects_in_region(detected_boxes, region, counted_objects)

            # 保存结果到列表
            for box in boxes:
                if box.id is not None:
                    all_results.append([
                        frame_idx,
                        int(box.id),
                        *box.xywh[0].tolist(),
                        box.conf.item()
                    ])

            # 在帧上绘制计数信息
            cv2.putText(frame, f'Objects in region: {count}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if frame_idx % frame_step == 0:
            output_frame = frame
            if blur_kernel > 1:
                output_frame = cv2.GaussianBlur(output_frame, (blur_kernel, blur_kernel), 0)
            if save_width != video_width or save_height != video_height:
                output_frame = cv2.resize(output_frame, (save_width, save_height), interpolation=cv2.INTER_AREA)
            out_video.write(output_frame)

    # 保存结果到文件
    if all_results:
        with open(output_txt_path, 'w') as f:
            for line in all_results:
                f.write(' '.join(map(str, line)) + '\n')
        print(f"跟踪结果已保存至：{output_txt_path}")

    # 释放资源
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"处理完成！输出文件保存在：{output_folder}")

    
