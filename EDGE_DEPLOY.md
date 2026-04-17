# Litchi-YOLO Edge Deployment

This project can be deployed on NVIDIA edge devices such as Jetson by exporting the detector to a TensorRT `engine`
file and keeping the tracking and counting logic in Python.

## 1. Files to copy to the edge device

Copy at least the following files and folders:

- `deploy_edge_tracking.py`
- `export_litchi_engine.py`
- `region_count.py`
- `ultralytics/`
- `models/yolov11-litchi.pt` or `runs/train/model/weights/best.pt`

If you already exported the engine on the target device, also keep:

- `models/yolov11-litchi.engine`

## 2. Export TensorRT engine

Recommended: export on the target edge device.

```bash
python export_litchi_engine.py \
  --weights models/yolov11-litchi.pt \
  --imgsz 1024 \
  --device 0 \
  --batch 1 \
  --workspace 4 \
  --half
```

For INT8 export:

```bash
python export_litchi_engine.py \
  --weights models/yolov11-litchi.pt \
  --imgsz 1024 \
  --device 0 \
  --batch 1 \
  --workspace 4 \
  --int8 \
  --data your_dataset.yaml
```

## 3. Run edge tracking and counting

Video file:

```bash
python deploy_edge_tracking.py \
  --model models/yolov11-litchi.engine \
  --source video/litchi1.mp4 \
  --tracker ultralytics/cfg/trackers/botsort.yaml \
  --imgsz 1024 \
  --device 0 \
  --save-video
```

USB camera:

```bash
python deploy_edge_tracking.py \
  --model models/yolov11-litchi.engine \
  --source 0 \
  --device 0 \
  --show
```

RTSP stream:

```bash
python deploy_edge_tracking.py \
  --model models/yolov11-litchi.engine \
  --source rtsp://username:password@ip:554/stream \
  --device 0 \
  --show
```

Custom counting region:

```bash
python deploy_edge_tracking.py \
  --model models/yolov11-litchi.engine \
  --source video/litchi1.mp4 \
  --region-json '{"x1":100,"y1":80,"x2":900,"y2":650}'
```

## 4. Output

The deployment script writes:

- `runs/edge_deploy/edge_tracking_result.mp4` when `--save-video` is enabled
- `runs/edge_deploy/edge_tracking_summary.json`

The summary contains:

- `end_to_end_fps`
- `avg_latency_ms`
- `counted_objects_total`
- `tracked_boxes_total`

## 5. Important notes

- TensorRT `engine` files are highly dependent on the GPU architecture, TensorRT version, CUDA version, and driver.
- The safest workflow is to export the `engine` directly on the target edge device.
- The detector runs through TensorRT, while the tracker and counting logic still run in Python.
