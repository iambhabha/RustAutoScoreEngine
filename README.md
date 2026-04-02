# 🎯 Rust Auto Score Engine — High-Performance Dart Scorer

<div align="center">

[![Rust](https://img.shields.io/badge/Rust-1.75+-black?style=for-the-badge&logo=rust)](https://www.rust-lang.org)
[![Burn](https://img.shields.io/badge/Framework-Burn-orange?style=for-the-badge)](https://burn.dev)
[![WGPU](https://img.shields.io/badge/Hardware-WGPU-blue?style=for-the-badge)](https://wgpu.rs)
[![WASM](https://img.shields.io/badge/Web-WASM-purple?style=for-the-badge&logo=webassembly)](https://webassembly.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A high-concurrency, real-time dart scoring and board analytics engine architected in Rust.**

*The professional Rust port of the original [Dart-Vision](https://github.com/iambhabha/Dart-Vision) project.*

![Dashboard Preview](image.png)

</div>

---

## ✨ What Does This Do?

The Rust Auto Score Engine uses a deep neural pipeline to:

1.  📷 **Capture/Upload** a dartboard image (800x800).
2.  🔍 **Detect** dart positions and 4 calibration corners in under 50ms.
3.  📐 **Reconstruct** board areas even if corners are obscured by players.
4.  🧮 **Calculate** the exact score based on BDO professional standards.
5.  📊 **Display** a professional dashboard with SVG overlays and confidence metrics.

---

## 🌐 Live Demonstrations

Experience the engine directly in your browser or via cloud hosting:

| Platform | Link | Technology |
| :--- | :--- | :--- |
| **WASM Site** | [🔗 Open Demo](https://iambhabha.github.io/RustAutoScoreEngine/) | Rust + WebAssembly (Local Inference) |
| **HF Space** | [🔗 Open Space](https://huggingface.co/spaces/bhabha-kapil/RustAutoScoreEngine) | Docker + Axum (Cloud Inference) |

---

## 🛠️ Performance & CLI

The engine is optimized for sub-millisecond latency on modern GPUs.

| Command | Category | Key Features |
| :--- | :--- | :--- |
| `cargo run --release -- gui` | **Dashboard** | Full Web UI (8080), Live SVG, Real-time Analysis |
| `cargo run --release -- train` | **Training** | Adam Optimizer, DIOU Loss, Auto-Checkpointing |
| `cargo run --release -- test <path>` | **Diagnostics** | Raw Coordinate Reporting, Confidence Metrics |

> **Note:** For maximum performance, always use the `--release` flag during execution.

---

## 🚀 Installation & Setup

### 1. Prerequisites
- **Rust Toolchain:** Stable channel (1.75+).
- **GPU Drivers:** Support for Vulkan, Metal, or DX12 (via WGPU).

### 2. Quick Deployment
```bash
# Clone Repo
git clone https://github.com/iambhabha/RustAutoScoreEngine.git
cd RustAutoScoreEngine

# Ensure weights are present
# Place 'model_weights.bin' in the root directory

# Launch Dashboard
cargo run --release -- gui
```

---

## 🏋️ Neural Training Ecosystem

Follow these steps to train the model on your own hardware using the **Burn** framework.

### 1. Dataset Preparation
Download the **[Official IEEE DeepDarts Collection](https://ieee-dataport.org/open-access/deepdarts-dataset)** (16K+ images) and extract them:
```bash
# Extract into dataset/800 folder
unzip images.zip -d dataset/800/
```

### 2. Config & Labels
Ensure `dataset/labels.json` exists with mappings for:
- **Class 0:** Dart Tip
- **Class 1-4:** Calibration Corners

### 3. Execution
```bash
cargo run --release -- train
```
> 💡 **VRAM Optimization:** The engine is optimized for as little as **3.3GB VRAM**, making it training-ready for consumer laptops and GPUs.
>
> ⚙️ **GPU Tuning:** All performance and accuracy settings are now in [src/config.rs](src/config.rs). See [SETTINGS.md](SETTINGS.md) for detailed instructions on how to optimize for 100GB vs RTX 5080 VRAM.

---

## 🤖 Auto-Labeling Ecosystem (AI-Powered Generation)

Need to generate a large labeled dataset quickly? Use our built-in auto-labelers to detect corners and darts automatically from raw images.

### 🐍 Python Auto-Labeler (YOLOv8 Based)
This uses a pre-trained `model.pt` to generate `labels_auto.json` and high-quality visuals.
```bash
python tools/labeler_v8.py
```
- **Clean Visuals:** Draws sharp 12x12 boxes with color-coded labels (C1-C4, D).
- **Customizable:** Adjust box size or colors in the `generate()` function using OpenCV.

### 🦀 Rust Auto-Labeler (ONNX Based)
A pure Rust implementation for ultra-fast, dependency-free dataset labeling using `model.onnx`.
```bash
cargo run --bin autolabel
```
- **Pixel-Art Labels:** Custom "D" and "C1-C4" renderer for clear verification without font dependencies.
- **Fixed Boxes:** Uses synchronized 12x12 boxes to match the Python output exactly.
- **Legend:** Prints a color guide (Red=C1, Green=C2, etc.) directly in your terminal.

---

## 📁 Project Structure

```text
RustAutoScoreEngine/
├── src/
│   ├── main.rs            ← Entry point (CLI Handler)
│   ├── bin/
│   │   └── autolabel.rs   ← Rust Auto-Labeler (ONNX)
│   ├── model.rs           ← YOLOv4-tiny (Burn Implementation)
│   ├── server.rs          ← Axum Web Server & API
│   ├── train.rs           ← Training Loop & Optimizers
│   ├── scoring.rs         ← BDO Standard Mathematics
│   └── wasm_bridge.rs     ← Browser Bindings
├── tools/
│   └── labeler_v8.py      ← Python Auto-Labeler (YOLOv8)
├── static/
│   └── index.html         ← Professional Dashboard UI
├── dataset/
│   ├── 800/               ← Training Images
│   └── labels.json        ← Ground Truth Metadata
├── model_weights.bin      ← Trained Neural Weights
├── Dockerfile             ← Hugging Face Space Config
├── Cargo.toml             ← Dependencies & Build Config
├── SETTINGS.md            ← GPU & Training Optimization Guide
└── README.md              ← Project Documentation
```

---

## 📐 Board Geometry (BDO Standard)

| Region | Radius (mm) |
| :--- | :--- |
| **Full Board** | 225.5 |
| **Double Ring** | 170.0 |
| **Treble Ring** | 107.4 |
| **Outer Bull** | 15.9 |
| **Bullseye** | 6.35 |

---

## 📄 Research & Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{mcnally2021deepdarts,
  title     = {DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera},
  author    = {McNally, William and Narasimhan, Kanav and Guttikonda, Srinivasu and Yampolsky, Alexander and McPhee, John and Wong, Alexander},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVSports)},
  year      = {2021}
}
```

---

<div align="center">
Made with ❤️ by bhabha-kapil — RustAutoScoreEngine
</div>
