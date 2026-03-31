/// 🎯 GLOBAL PERFORMANCE & ACCURACY CONFIGURATION
/// Use this file to tune the engine for your specific GPU (RTX 5080, A100/H100, etc.)

// --- 🖥️ HARDWARE SETTINGS (GPU/CPU) ---

/// GPU Device Selection
/// - WgpuDevice::default(): Auto-detect (Recommended)
/// - WgpuDevice::DiscreteGpu(0): Force first dedicated GPU
pub const GPU_DEVICE_ID: usize = 0;

/// Number of Parallel I/O Workers for Data Loading
/// - 4: Standard Gaming PC
/// - 8-12: High-end Gaming PC (RTX 5080)
/// - 24-32: Server-grade hardware (H100/A100)
pub const NUM_WORKERS: usize = 4; // Reduced to 4 for stability debugging

// --- 🏋️ TRAINING HYPERPARAMETERS (For 99.994% Accuracy) ---

/// Total Training Epochs
/// - 50: Fast training (~90% accuracy)
/// - 150-200: Deep training (Necessary for 99.994% target)
pub const NUM_EPOCHS: usize = 200;

/// Training Batch Size (THE MOST IMPORTANT SETTING)
/// - 1-4: Low-end GPUs (4GB VRAM)
/// - 16-32: RTX 3080 / 4080 / 5080 (16-24GB VRAM)
/// - 128-512: A100 / H100 (40GB-100GB VRAM)
pub const BATCH_SIZE: usize = 1; // Reduced to 1 to debug the 71GB allocation error

/// Base Learning Rate
pub const LEARNING_RATE: f64 = 1e-3;

/// Warmup Epochs (Gradual increase for stability)
pub const WARMUP_EPOCHS: usize = 5;

// --- 📐 LOSS FUNCTION WEIGHTS (Precision Tuning) ---

/// Weight for Objectness Loss (BCE)
/// (How strongly the model focuses on *finding* a point)
pub const WEIGHT_OBJ_LOSS: f32 = 50.0;

/// Weight for Class Classification Loss
/// (Distinguishing between Dart Tip and Corners)
pub const WEIGHT_CLASS_LOSS: f32 = 25.0;

/// Weight for XY Coordinate Precision (CRITICAL FOR ACCURACY)
/// (Higher = More precise pixel detection. Use 80+ for 99.994% accuracy)
pub const WEIGHT_XY_LOSS: f32 = 80.0;

/// Weight for Width/Height Precision
pub const WEIGHT_WH_LOSS: f32 = 5.0;

// --- 💾 STORAGE & CHECKPOINTS ---

/// File path to save/load weights
pub const MODEL_WEIGHTS_FILE: &str = "model_weights";

/// Periodic Save Interval (Batches)
/// - 100: Save frequently (Safe)
/// - 500: Save less often (Fastest throughput on high-end GPUs)
pub const SAVE_INTERVAL_BATCHES: usize = 500;

// --- 📊 VALIDATION SETTINGS ---

/// Percentage of dataset to use for validation (0.1 = 10%)
pub const VALIDATION_SPLIT: f64 = 0.1;
