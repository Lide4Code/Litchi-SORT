# Litchi-SORT

Official implementation of **Litchi-SORT: Overcoming Occlusion and Motion Instability for Accurate Low-Altitude UAV-Based Litchi Tracking and Counting**.

This repository releases the **code** and **pretrained weights** for our litchi detection, tracking, and counting pipeline. The paper is currently under review. The dataset will be released progressively.

## Overview

Real-time multi-object tracking in large-scale UAV video streams is challenging due to dense clustering, severe occlusion, drastic scale variation, and motion blur. To address these issues, we propose **Litchi-SORT**, a robust tracking and counting framework tailored for low-altitude UAV-based litchi monitoring.

Compared with the BoT-SORT baseline, Litchi-SORT improves robustness through:

- A region-based counting method with a frame threshold to suppress transient tracking errors
- A Kalman Filter with Dynamic Adaptive Noise Covariance for more stable state estimation
- A multi-model motion filtering and trajectory smoothing strategy for improved prediction
- EIoU-based data association for dense-object scenarios

## Abstract

Real-time multi-object tracking in large-scale Unmanned Aerial Vehicle (UAV) video streams is a computationally intensive task, where algorithm performance is critical. State-of-the-art trackers like BoT-SORT suffer from performance degradation, including frequent track fragmentation and identity switches (IDSW), when faced with dense clustering, severe occlusion, drastic scale variations, and motion blur. These challenges severely compromise the accuracy of subsequent analysis, such as object counting. To address these performance bottlenecks, we propose Litchi-SORT, a robust tracking and counting algorithm tailored for complex UAV-based monitoring. Litchi-SORT enhances the BoT-SORT computational framework through four key algorithmic innovations: 1) a novel region-based counting method with a frame threshold to mitigate transient tracking errors; 2) a Kalman Filter with Dynamic Adaptive Noise Covariance for improved state estimation under motion noise; 3) a multi-model motion filtering and trajectory smoothness strategy for more accurate prediction; and 4) replacing the standard IoU metric with EIoU for superior data association in dense object scenarios. Validated on a challenging custom UAV dataset comprising 20 videos at 60 FPS, Litchi-SORT significantly outperforms the BoT-SORT baseline by improving Multi-Object Tracking Accuracy (MOTA) from 75.28 to 77.26, reducing average IDSW from 114.1 to 85.1, and achieving a robust counting accuracy of 91.05%. This work contributes an advanced and efficient algorithm for processing demanding aerial data, demonstrating improved performance and reliability for systems requiring high-accuracy object tracking.

## Demo

### Detection Result

![Detection Result](image/test_result.jpg)

### Tracking Result

Click the GIF preview below to open the full demo video:

[![Tracking Demo GIF](image/litchi9_preview.gif)](video/litchi9_result.mp4)

- [Download / Open the full MP4 demo](video/litchi9_result.mp4)

## Performance Comparison

| Method | MOTA (%) ↑ | IDF1 (%) ↑ | HOTA (%) ↑ | IDSW ↓ | TF ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SORT | 64.13 | 83.24 | 73.68 | 157.2 | 85.9 |
| DeepSORT | 68.11 | 85.13 | 76.62 | 145.7 | 84.7 |
| StrongSORT | 71.53 | 86.76 | 79.14 | 129.4 | 83.6 |
| ByteTrack | 74.76 | 87.29 | 81.02 | 118.5 | 82.6 |
| BoT-SORT | 75.28 | 87.52 | 81.40 | 114.1 | 82.5 |
| **Litchi-SORT** | **77.26** | **89.09** | **83.57** | **85.1** | **81.2** |

## Installation

### 1. Clone the repository

```bash
git clone <your-github-repo-url>
cd litchi_yolo_v2
```

### 2. Create a Python environment

```bash
conda create -n litchi-sort python=3.10 -y
conda activate litchi-sort
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Notes:

- A CUDA-capable GPU is strongly recommended for tracking on UAV videos
- Released model files are provided under `models/`
- `track.py` will first try `runs/train/model/weights/best.pt`, and automatically fall back to `models/yolov11-litchi.pt`

## Quick Start

The easiest way to reproduce the tracking result is to run `track.py`.

### Run tracking and counting

```bash
python track.py \
  --input video/litchi9.mp4 \
  --output runs/track/litchi9_demo \
  --save-scale 0.5 \
  --save-fps 30 \
  --blur-kernel 3
```

This command will generate:

- A tracking video: `runs/track/litchi9_demo/litchi9_result.mp4`
- A tracking result text file: `runs/track/litchi9_demo/litchi9.txt`

### Output arguments

`track.py` supports several useful options for export:

- `--save-scale`: downscale the saved video to reduce file size
- `--save-fps`: change the saved video FPS, default is to follow the source video
- `--blur-kernel`: lightly blur the saved video to reduce bitrate and make playback smoother
- `--codec`: choose output codec, such as `auto`, `avc1`, `H264`, or `mp4v`

### Optional image detection example

```bash
python detect.py \
  --input image/litchi.jpg \
  --output detection_results \
  --folder demo
```

## Repository Structure

- `track.py`: main litchi tracking and counting entry point
- `detect.py`: image detection demo
- `models/`: released model files in `.pt`, `.onnx`, and `.engine` formats
- `image/`: demo images for README and testing
- `video/`: demo videos and tracking results

## Open-Source Plan

- Released now: code and pretrained weights
- Coming next: dataset and annotations will be released progressively

## Citation

If this repository is useful for your research, please cite our work. A formal BibTeX entry will be added after publication.

```text
Litchi-SORT: Overcoming Occlusion and Motion Instability for Accurate Low-Altitude UAV-Based Litchi Tracking and Counting
```

## License

This repository is released under the license provided in [LICENSE](LICENSE).
