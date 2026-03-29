use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use rust_auto_score_engine::args::{AppArgs, Command};
use rust_auto_score_engine::server::start_gui;
use rust_auto_score_engine::tests::test_model;
use rust_auto_score_engine::train::{train, TrainingConfig};

fn main() {
    let app_args = AppArgs::parse();
    let device = WgpuDevice::default();

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
            println!("[Burn-DartVision] Starting Full Project Training...");
            let dataset_path = "dataset/labels.json";

            let config = TrainingConfig {
                num_epochs: 50,
                batch_size: 1,
                lr: 1e-3,
            };

            train::<burn::backend::Autodiff<Wgpu>>(device, dataset_path, config);
        }
    }
}
