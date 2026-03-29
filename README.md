---
title: Rust Auto Score Engine
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Rust Auto Score Engine

A high-performance dart scoring system architected in Rust, utilizing the Burn Deep Learning framework. This engine is the high-concurrency Rust port of the original Python-based [Dart-Vision](https://github.com/iambhabha/Dart-Vision) project.

![Dashboard Preview](image.png)

---

## Live Demo (No Server Required)

The entire neural engine can run directly in your browser using **WebAssembly (WASM)**. No installation or heavy server is required.

**Try it here (WASM):** [https://iambhabha.github.io/RustAutoScoreEngine/](https://iambhabha.github.io/RustAutoScoreEngine/)

**Try it here (Hugging Face Space):** [https://huggingface.co/spaces/bhabha-kapil/RustAutoScoreEngine](https://huggingface.co/spaces/bhabha-kapil/RustAutoScoreEngine)

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
3. For local dashboard, run the `gui` command.
4. For custom training, follow the detailed guide below.

---

## 🚀 How to Train Your Own Model (Step-by-Step Guide)

If you want to train the engine from scratch or with your own custom dartboard dataset, follow these exact steps:

### **Step 1: Download the Official Dataset**
1.  **Source:** Go to the [Official IEEE DeepDarts Collection (16K+ Images)](https://ieee-dataport.org/open-access/deepdarts-dataset).
2.  **Download:** Download the `.zip` file containing high-resolution board imagery.
3.  **Extraction:** Extract the images and place them in the project folder at:
    `dataset/800/`

### **Step 2: Setup Labels and Configuration**
1.  **Metadata:** Ensure you have a `labels.json` file in the `dataset/` folder.
2.  **Format:** Each entry should map an image filename to its calibration corners (Classes 1-4) and dart keypoints (Class 0).
3.  **Pre-processing:** If your images are larger than 800x800, the engine will automatically resize them during training, but 800x800 is recommended for performance.

### **Step 3: Initialize Neural Training**
Launch the high-concurrency training engine via terminal:
```bash
cargo run --release -- train
```
*Tip: Ensure your GPU drivers are updated (Vulkan/Metal/DX12 are supported via WGPU).*

### **Step 4: Real-time Checkpointing**
-   **Auto-Save:** The training loop automatically updates `model_weights.bin` at the end of every epoch.
-   **Stability:** If training is interrupted, simply restart the command to pick up from the latest loss metrics.
-   **VRAM Usage:** Optimized for ~3.3GB VRAM, allowing training on single consumer-grade GPUs.

### **Step 5: Verify Your Neural Weights**
Once you have trained the model (minimum 20-50 epochs recommended), verify the detection accuracy using the built-in GUI:
```bash
cargo run --release -- gui
```
The Dashboard will launch on **http://localhost:8080**, where you can drop new images to see real-time coordinate recovery and scoring!

---

## Advanced Architecture and Optimization

### 1. Distance-IOU (DIOU) Loss Implementation
Utilizing DIOU Loss ensures stable training and faster convergence for small objects like dart tips by calculating intersection over union alongside center distance.

### 2. Deep-Dart Symmetry Engine
If a calibration corner is obscured, the system invokes a symmetry-based recovery algorithm to reconstruct the board area without recalibration.

### 3. Memory & VRAM Optimization
Optimized to handle 800x800 resolution training on consumer GPUs by efficiently detaching the Autodiff computation graph during logging cycles (Usage: ~3.3GB VRAM).

---

## Resources and Research

### Scientific Publications
- **arXiv Project (2105.09880):** [DeepDarts Neural Network Paper](https://arxiv.org/abs/2105.09880)
- **Original Project:** [iambhabha/Dart-Vision](https://github.com/iambhabha/Dart-Vision)

### Source Materials
- **Dataset (IEEE Dataport):** [Official DeepDarts Collection (16K+ Images)](https://ieee-dataport.org/open-access/deepdarts-dataset)
- **Framework (Burn):** [Burn Deep Learning Documentation](https://burn.dev/book/)

---
*Distributed under the terms of the project's license. Built for the global darts technology ecosystem.*
