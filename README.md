# 🎯 DartVision AI - Rust AutoScore Engine

A high-performance dart scoring system built with **Rust** and the **Burn** Deep Learning framework. This project is a port of the original YOLOv4-tiny based DartVision, optimized for speed and safety.

![DartVision Dashboard](https://raw.githubusercontent.com/iambhabha/RustAutoScoreEngine/main/docs/dashboard.png)

## 🚀 Quick Start (GUI Dashboard)
The project comes with pre-trained weights (`model_weights.bin`). You can start the professional dashboard immediately:

```bash
cargo run --release -- gui
```
Then open: **[http://127.0.0.1:8080](http://127.0.0.1:8080)**

## 📈 Training
To train the model on your own dataset (requires `dataset/labels.json` and images):

```bash
cargo run --release -- train
```
*Note: The model saves checkpoints every 100 batches. You can stop and resume training anytime.*

## 🔬 Testing
To run a single image inference and see the neural mapping results:

```bash
cargo run --release -- test <path_to_image>
```

## ✨ Features
- **Neural Mapping:** Real-time detection of darts and 4 calibration corners.
- **Smart Scoring:** Automatic coordinate reconstruction and BDO standard scoring.
- **Reliability Checks:** GUI displays per-point confidence percentages (CAL Sync) to ensure accuracy.
- **GPU Accelerated:** Powered by `WGPUDevice` and `Burn` for ultra-fast inference.

## 🛠 Project Structure
- `src/model.rs`: YOLOv4-tiny architecture in Burn.
- `src/loss.rs`: DIOU Loss + Objectness + Class entropy implementation.
- `src/server.rs`: Axum-based web server for the GUI.
- `static/index.html`: Premium Glassmorphism interface with SVG overlays.

---
*Created by [iambhabha](https://github.com/iambhabha)*
