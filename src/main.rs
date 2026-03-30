use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use rust_auto_score_engine::args::{AppArgs, Command};
use rust_auto_score_engine::server::start_gui;
use rust_auto_score_engine::tests::test_model;
use rust_auto_score_engine::train::{train, TrainingConfig};
use rust_auto_score_engine::config as cfg; // Centralized Config

fn main() {
    let app_args = AppArgs::parse();
    
    // Choose GPU device based on config (0 = Discrete GPU, default otherwise)
    let device = if cfg::GPU_DEVICE_ID == 0 {
        WgpuDevice::default()
    } else {
        WgpuDevice::DiscreteGpu(cfg::GPU_DEVICE_ID - 1)
    };

    match app_args.command {
        Command::Gui => {
            println!("[Burn-DartVision] Starting Professional Dashboard...");
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(start_gui(device));
        }
        Command::Test { img_path } => {
            test_model(device, &img_path);
        }
        Command::Train => {
            println!("[Burn-DartVision] Global Project Training (Target: 99.994% Accuracy)...");
            
            // --- ⚙️ USING CENTRALIZED CONFIGURATION FROM src/config.rs ---
            let dataset_path = "dataset/labels.json";

            let config = TrainingConfig {
                num_epochs: cfg::NUM_EPOCHS,
                batch_size: cfg::BATCH_SIZE,
                lr: cfg::LEARNING_RATE,
            };

            train::<burn::backend::Autodiff<Wgpu>>(device, dataset_path, config);
        }
    }
}
