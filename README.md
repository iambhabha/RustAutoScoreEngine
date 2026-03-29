# Rust Auto Score Engine

A high-performance dart scoring system architected in Rust, utilizing the Burn Deep Learning framework. This engine is the high-concurrency Rust port of the original Python-based [Dart-Vision](https://github.com/iambhabha/Dart-Vision) project.

![Dashboard Preview](image.png)

---

## Project Origin

This system is an optimized re-implementation of the **[Dart-Vision](https://github.com/iambhabha/Dart-Vision)** repository. While the original project provided the foundational neural logic in Python/PyTorch, this engine focuses on:
- **Performance:** Sub-millisecond tensor operations using the Burn framework.
- **Safety:** Eliminating runtime overhead and ensuring thread-safe inference.
- **Modern UI:** Transitioning from local scripts to a professional Glassmorphism web dashboard.

---

## Technical Overview

The engine implements a multi-stage neural pipeline designed for millisecond-latency inference and robust coordinate recovery.

### Neural Pipeline Workflow
- **Input:** 800x800 RGB Image Frame.
- **Mapping:** YOLOv4-tiny backbone extracts point-of-interest features.
- **Verification:** Keypoint confidence filtering and objectness thresholds.
- **Symmetry Recovery:** Mathematical reconstruction of obscured calibration points.
- **Scoring:** BDO standard coordinate mapping for final point calculation.

---

## CLI Reference Guide

| Command | Action | Key Features |
| :--- | :--- | :--- |
| `cargo run --release -- gui` | Start Dashboard | Web UI (8080), Live SVG Overlays, Confidence Diagnostics |
| `cargo run --release -- train` | Begin Training | Adam Optimizer, DIOU Loss, Auto-Checkpointing |
| `cargo run --release -- test <path>` | Direct Inference | Raw Coordinate Reporting, Confidence Analysis |

---

## Installation and Setup

### Prerequisites
- **Rust Toolchain:** Stable channel (Latest).
- **GPU Driver:** Support for Vulkan, Metal, or DirectX 12 (via WGPU).
- **Hard Drive Space:** Minimum 1GB (excluding dataset).

### Initial Setup
1. Clone the repository.
2. Ensure `model_weights.bin` is present in the root directory.
3. To test the GUI immediately, run the `gui` command.
4. For custom training, place your images in `dataset/800/` and configuration in `dataset/labels.json`.

---

## Advanced Architecture and Optimization

### 1. Distance-IOU (DIOU) Loss Implementation
Our implementation moves beyond standard Mean Squared Error. By utilizing DIOU Loss, the engine optimizes for:
- Overlap area between prediction and target.
- Euclidean distance between the central points.
- Geometric consistency of the dart point shape.

### 2. Deep-Dart Symmetry Engine
If a calibration corner is missing or obscured by a dart or observer, the system invokes a symmetry-based recovery algorithm. By calculating the centroid of the remaining points and applying rotational offsets, the board coordinates are maintained without recalibration.

### 3. Memory & VRAM Optimization
The training loop is architected to detach the Autodiff computation graph during logging cycles. This reduces VRAM consumption from an unoptimized 270GB down to approximately 3.3GB per training sample at 800x800 resolution.

---

## Resources and Research

This project is built upon advanced research in the computer vision and darts community:

### Scientific Publications
- **arXiv Project (2105.09880):** [DeepDarts: Neural Network for Coordinate Reconstruction](https://arxiv.org/abs/2105.09880)
- **Darknet Research:** [YOLOv4-tiny Implementation and Paper](https://pjreddie.com/darknet/yolo/)

### Source Materials
- **Original Project:** [iambhabha/Dart-Vision](https://github.com/iambhabha/Dart-Vision)
- **Dataset (IEEE Dataport):** [Official DeepDarts Collection (16K+ Images)](https://ieee-dataport.org/open-access/deepdarts-dataset)
- **Framework (Burn):** [Burn Deep Learning Documentation](https://burn.dev/book/)

---
*Distributed under the terms of the project's license. Built for the global darts technology ecosystem.*
