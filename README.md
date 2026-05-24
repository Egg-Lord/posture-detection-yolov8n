# Posture Detection using YOLOv8

This project is a deep learning-based posture detection system using YOLOv8.

The model detects three posture classes:
- Proper
- Slouch
- Leaning Back

## Features

- Real-time posture detection
- Image upload testing
- Confidence score display
- Bounding box visualization
- Streamlit web interface

## Dataset

Dataset was:
- Annotated using Roboflow
- Augmented with:
  - Horizontal Flip
  - Rotation (-15° to +15°)
  - Brightness Adjustment (-15% to +15%)

### Dataset Split

| Set | Images |
|---|---|
| Train | 3156 |
| Validation | 301 |
| Test | 150 |

## Model

- YOLOv8n
- Image Size: 512x512
- Epochs: 100
- Batch Size: 16
- GPU: Tesla T4

## Performance Metrics

| Metric | Score |
|---|---|
| Precision | 0.99 |
| Recall | 0.99 |
| mAP50 | 0.99 |
| mAP50-95 | 0.82 |

## Installation

```bash
pip install -r requirements.txt
