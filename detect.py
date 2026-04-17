import warnings
import argparse
from ultralytics import YOLO
import os
import json

# python detect.py --input images/img1.jpg,images/img2.jpg --output detection_results --folder exp1

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--input', type=str, required=True, help='Input image path(s), separated by commas')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--folder', type=str, required=True, help='Experiment folder name')
    return parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    args = parse_arguments()
    input_images = args.input.split(',')

    # 加载模型
    model = YOLO('models/yolov11-litchi.pt')

    # 执行预测
    results = model.predict(
        source=input_images,
        imgsz=640,
        project=args.output,
        name=args.folder,
        save=True,
        exist_ok=True # 允许覆盖已有文件
    )

    # 构造 summary 数据
    summary_data = []
    for i, result in enumerate(results):
        im_path = input_images[i]
        im_name = os.path.basename(im_path)

        boxes = result.boxes
        count = len(boxes)
        if count > 0 and boxes.conf is not None:
            confidence = round(float(boxes.conf.mean()) * 100, 2)
        else:
            confidence = 0.0

        summary_data.append({
            "image_name": im_name,
            "count": count,
            "confidence": confidence
        })

    # 保存为 JSON 文件
    summary_path = os.path.join(args.output, args.folder, 'detection_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
