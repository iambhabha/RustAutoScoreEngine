# RustAutoScoreEngine
### High-Performance AI Dart Scoring Powered by Rust & Burn

<div align="center">

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)
[![Burn](https://img.shields.io/badge/Burn-AI--Framework-red?style=for-the-badge)](https://burn.dev/)
[![WGPU](https://img.shields.io/badge/Backend-WGPU%20/%20Cuda-blue?style=for-the-badge)](https://github.com/gfx-rs/wgpu)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

**A professional-grade, real-time dart scoring engine built entirely in Rust.**

Using the **Burn Deep Learning Framework**, this project achieves sub-millisecond inference and high-precision keypoint detection for automatic dart game tracking. The model optimization pipeline is built using modern Rust patterns for maximum safety and performance.

</div>

---

## Features

- **Optimized Inference**: Powered by Rust & WGPU for hardware-accelerated performance on Windows, Linux, and macOS.
- **Multi-Scale Keypoint Detection**: Enhanced YOLO-style heads for detecting dart tips and calibration corners.
- **BDO Logic Integrated**: Real-time sector calculation based on official board geometry and calibration symmetry.
- **Modern Web Dashboard**: Axum-based visual interface to monitor detections, scores, and latency in real-time.
- **Robust Calibration**: Automatic symmetry estimation to recover missing calibration points.

---

## Dataset and Preparation
The model is trained on the primary dataset used for high-precision dart detection.

- **Download Link**: [Dataset Resources (Google Drive)](https://drive.google.com/file/d/1ZEvuzg9zYbPd1FdZgV6v1aT4sqbqmLqp/view?usp=sharing)
- **Resolution**: 800x800 pre-cropped high-resolution images.
- **Structure**: Organize your data in the `dataset/800/` directory following the provided `labels.json` schema.

---

## Installation

### 1. Install Rust
If you do not have Rust installed, use the official installation script:

```bash
# Official Installation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Clone and Build
```bash
git clone https://github.com/iambhabha/RustAutoScoreEngine.git
cd RustAutoScoreEngine
cargo build --release
```

---

## Quick Start Guide

### Step 1: Training the AI Model
To optimize the neural network for your local environment, run the training mode:
```bash
# Starts the training cycle (Configured for 50 Epochs)
cargo run
```
*Tip: Allow the loss to converge below 0.05 for optimal results.*

### Step 2: Running the Dashboard
After training is complete (generating the `model_weights.bin` file), launch the testing interface:
```bash
# Starts the Axum web server
cargo run -- gui
```
**Features:**
- **Image Upload**: Test local image samples via the dashboard.
- **Point Visualization**: Inspect detected calibration points and dart locations.
- **Automatic Scoring**: Instant sector calculation and latency reporting.

---

## Mobile Deployment

This engine is built on Burn, supporting multiple paths for Android and iOS integration:

### Path A: Native Rust
Package the engine as a library for direct hardware-accelerated execution on mobile targets.
- **Backend**: burn-wgpu with Vulkan (Android) or Metal (iOS).
- **Integration**: JNI (Android) or FFI (iOS) calls from native code.

### Path B: Weight Migration to TFLite/ONNX
- **TFLite**: Use the companion export scripts to generate a TensorFlow Lite bundle.
- **ONNX**: Utilize ONNX Runtime (ORT) for high-performance cross-platform execution.

---

## Hardware Optimization

This engine is optimized for GPU execution using the WGPU backend. Depending on your specific hardware, you may need to adjust the training intensity:

### GPU VRAM Management
If you encounter **Out-of-Memory (OOM)** errors during training, you should reduce the **Batch Size**.

- **Where to change**: Open `src/main.rs` and modify the `batch_size` parameter.
- **Recommendations**:
  - **4GB VRAM**: Batch Size 1 (Safe default)
  - **8GB VRAM**: Batch Size 4
  - **12GB+ VRAM**: Batch Size 8
  - **RTX 5080 High-End**: Batch Size 16 (Optimal for ultra-fast convergence)
- **Impact**: Larger batch sizes provide more stable gradients but require exponentially more VRAM.

---

## Technical Status and Contributing

> [!IMPORTANT]
> This project is currently in the experimental phase. We are actively refining the coordinate regression logic to ensure maximum precision across diverse board angles.

**Current Priorities:**
- Enhancing offset regression stability.
- Memory optimization for low-VRAM devices.

**Contribution Guidelines:**
If you encounter a bug or wish to provide performance optimizations, please submit a Pull Request.

---

## Resources

- **Original Inspiration**: [Paper: Keypoints as Objects for Automatic Scorekeeping](https://arxiv.org/abs/2105.09880)
- **Model Training Resources**: [Download from Google Drive](https://drive.google.com/file/d/1ZEvuzg9zYbPd1FdZgV6v1aT4sqbqmLqp/view?usp=sharing)
- **Official Documentation Reference**: [IEEE Dataport Dataset](https://ieee-dataport.org/open-access/deepdarts-dataset)

---

<div align="center">
Made by the Rust AI Community
</div>
