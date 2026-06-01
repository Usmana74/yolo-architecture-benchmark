# Architectural Trade-offs in Single-Stage Object Detection: A Comparative Study of YOLOv7, YOLOv8n, and YOLOv9c

> **Do architectural differences between anchor-based and anchor-free single-stage detectors produce measurable, consistent speed–accuracy trade-offs under identical inference conditions?**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8%2Fv9-purple.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Usmana74/yolo-architecture-benchmark/blob/main/yolo_benchmark.ipynb)

---

## Overview

This repository presents a systematic empirical comparison of three prominent YOLO architectures representing three generations of single-stage object detection design:

| Model | Architecture Paradigm | Key Innovation |
|-------|----------------------|----------------|
| **YOLOv7** | Anchor-based | Extended efficient layer aggregation networks (E-ELAN) |
| **YOLOv8n** | Anchor-free (nano) | Decoupled head, task-aligned assignment, C2f module |
| **YOLOv9c** | Anchor-free + PGI | Programmable Gradient Information (PGI), GELAN backbone |

The central question: across three distinct architectural generations — anchor-based, anchor-free, and PGI-augmented anchor-free — do the design choices produce **measurable, consistent trade-offs** between inference speed, detection confidence, and detection volume?

---

## Key Findings

```
YOLOv7  → Fastest  (13.5 ms/image) · Most detections (56) · Avg confidence 62.4%
YOLOv8n → Balanced (37.5 ms/image) · 41 detections       · Avg confidence 61.8%
YOLOv9c → Highest confidence (67.5%) · 51 detections     · Slowest (61.6 ms/image)
```

**The Speed–Accuracy–Deployability Trilemma:** These results quantify a three-way trade-off across architectural generations. No single model dominates all three dimensions simultaneously — a result consistent with No Free Lunch theory applied to detection architectures.

- **Speed axis**: Anchor-based YOLOv7's dense prediction head, though computationally heavier in training, runs faster at inference on standard hardware — likely due to reduced post-processing overhead compared to decoupled anchor-free heads.
- **Confidence axis**: PGI in YOLOv9c improves gradient information preservation during deep network training, which translates to higher average detection confidence even on a small pilot set.
- **Deployability axis**: YOLOv8n's Ultralytics API offers the lowest integration overhead, with automatic mixed precision, export to ONNX/TensorRT, and single-line inference — a practical advantage orthogonal to raw benchmark numbers.

---

## Methodology

### Models & Weights
All models evaluated with **COCO-pretrained weights** (MS-COCO 80-class):
- YOLOv7: `yolov7.pt` — WongKinYiu official release
- YOLOv8n: `yolov8n.pt` — Ultralytics official
- YOLOv9c: `yolov9c.pt` — Ultralytics official

### Dataset
**COCO-2017 validation set** (pilot subset, n=10) downloaded via FiftyOne zoo.

> **Scope note**: This is a **pilot evaluation** on n=10 images. Confidence score is used as a detection quality proxy; it is **not equivalent to mAP@[0.5:0.95]**, the standard COCO metric. Full evaluation on the complete COCO-2017 validation set (5,000 images) with standard PASCAL VOC metrics is identified as necessary for statistically robust conclusions. See [Limitations & Future Work](#limitations--future-work).

### Metrics
| Metric | Definition | Limitation |
|--------|-----------|------------|
| Average confidence | Mean of all per-detection confidence scores across n images | Proxy only; does not account for false positives or IoU thresholds |
| Total detections | Count of boxes above conf=0.25 threshold | Sensitive to threshold; higher count ≠ higher precision |
| Inference latency | Wall-clock time per image (ms), averaged over n=10 runs | Single-run average; no statistical variance reported |

### Experimental Setup
- **Hardware**: Google Colab (NVIDIA T4 GPU, 16GB VRAM)
- **Confidence threshold**: 0.25 (identical across all models)
- **IoU threshold**: 0.45 (NMS, YOLOv7); Ultralytics default (v8/v9)
- **Input resolution**: 640×640 (all models)
- **Runtime isolation**: Models evaluated sequentially, same session

---

## Results

### Quantitative Summary

| Model | Avg Confidence (%) | Total Detections | Avg Latency (ms) | Architecture |
|-------|-------------------|-----------------|------------------|-------------|
| YOLOv7 | 62.44 | 56 | **13.5** | Anchor-based, E-ELAN |
| YOLOv8n | 61.81 | 41 | 37.5 | Anchor-free, C2f |
| **YOLOv9c** | **67.52** | 51 | 61.6 | Anchor-free + PGI, GELAN |

### Interpretation

**YOLOv7 (fastest)** — The anchor-based design with E-ELAN achieves 13.5 ms/image, 2.8× faster than YOLOv8n and 4.6× faster than YOLOv9c. This speed advantage likely reflects anchor prediction reuse and reduced post-processing in the dense prediction head. However, anchor-based designs require careful anchor configuration and are less flexible to novel object aspect ratios.

**YOLOv8n (balanced)** — The anchor-free nano variant delivers competitive confidence (61.8%) at moderate latency (37.5 ms). The Ultralytics API abstraction — single-line inference, automatic export pipelines, unified training/evaluation loop — contributes a **deployability advantage** not captured in raw benchmark numbers. For rapid prototyping and production integration, this is the practical choice.

**YOLOv9c (most confident)** — PGI's auxiliary reversible branch preserves gradient information across deep layers, which appears to translate into higher per-detection confidence (67.5%). This is consistent with the paper's claim that information bottleneck in deep networks degrades detection quality. The cost is a 61.6 ms latency — 4.6× slower than YOLOv7 — making it unsuitable for real-time (>30 fps) applications on T4-class hardware.

---

## Limitations & Future Work

### Current Limitations
1. **Sample size (n=10)**: Confidence and latency estimates have high variance on 10 images. Results are directionally informative but not statistically robust.
2. **Confidence ≠ mAP**: Average confidence score is a weak proxy. Standard evaluation requires mAP@0.5 and mAP@[0.5:0.95] on the full COCO validation set (5,000 images).
3. **Single hardware configuration**: Latency results are specific to T4 GPU on Colab. Results on CPU, edge hardware (Jetson Nano), or Apple Silicon would differ substantially.
4. **No variance reporting**: Single-run latency averages; standard deviation not reported.
5. **YOLOv7 patching required**: `torch.load` deprecation fix needed for PyTorch ≥ 2.0 — indicates maintenance debt in the YOLOv7 codebase.

### Future Work
- [ ] Full COCO-2017 validation evaluation (5,000 images) with mAP@[0.5:0.95]
- [ ] Per-category AP analysis (small/medium/large objects)
- [ ] Latency benchmarking across hardware targets: T4, V100, CPU, Jetson Nano
- [ ] YOLOv10 and RT-DETR inclusion for transformer-based comparison
- [ ] Input resolution sweep (320×320, 640×640, 1280×1280) — accuracy-latency Pareto curves
- [ ] Quantisation (INT8) and export (ONNX, TensorRT) pipeline evaluation

---

## Repository Structure

```
yolo-architecture-benchmark/
├── yolo_benchmark.ipynb      # Full Colab notebook — setup, inference, metrics, plots
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE
```

---

## Quickstart

```bash
git clone https://github.com/Usmana74/yolo-architecture-benchmark
cd yolo-architecture-benchmark
pip install -r requirements.txt
jupyter notebook yolo_benchmark.ipynb
```

**Recommended**: Run in Google Colab for GPU access and pre-configured environment.

---

## Dependencies

```
ultralytics>=8.0.0       # YOLOv8 and YOLOv9
torch>=2.0.0
torchvision>=0.15.0
opencv-python-headless>=4.8.0
numpy==1.23.5            # Required for YOLOv7 compatibility
fiftyone>=0.21.0         # COCO-2017 download via zoo
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
```

---

## Related Work

- Wang et al. (2022). *YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors.* CVPR 2023. [[arXiv](https://arxiv.org/abs/2207.02696)]
- Jocher et al. (2023). *Ultralytics YOLOv8.* [[GitHub](https://github.com/ultralytics/ultralytics)]
- Wang et al. (2024). *YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.* ECCV 2024. [[arXiv](https://arxiv.org/abs/2402.13616)]
- Lin et al. (2014). *Microsoft COCO: Common Objects in Context.* ECCV 2014. [[arXiv](https://arxiv.org/abs/1405.0312)]

---

## Citation

```bibtex
@misc{ahmad2026yolobenchmark,
  author = {Ahmad, Mohammad Usman},
  title  = {Architectural Trade-offs in Single-Stage Object Detection: YOLOv7 vs YOLOv8n vs YOLOv9c},
  year   = {2026},
  url    = {https://github.com/Usmana74/yolo-architecture-benchmark}
}
```

---

## Author

**Mohammad Usman Ahmad**
BS Computer Science (AI/CV), PMAS Arid Agriculture University, Pakistan · CGPA: 3.8/4.0

[GitHub](https://github.com/Usmana74) · [LinkedIn](https://linkedin.com/in/usman-ahmad-297b63262) · [PyPI — dataaudit](https://pypi.org/project/dataaudit/)
