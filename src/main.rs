mod config;
mod dataset;
mod model;
mod scoring;
mod train;
mod inference;
mod server;
mod yolo_ort;
mod tests;

use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};
use burn::backend::Wgpu;

#[tokio::main]
async fn main() {
    let device = WgpuDevice::default();
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("🔥 [DartVision-Pro] Native Rust Scoring Engine:");
        println!("   - train: Start training loop");
        println!("   - test <img_path>: RTX 5080 YOLO Native Test");
        println!("   - run <img_path>: Direct native image inference");
        println!("   - gui: Start Web-based GUI (Runs on PC Dashboard)");
        return;
    }

    match args[1].as_str() {
        "train" => {
            train::train::<Wgpu<WgpuRuntime, f32, i32>>(device);
        }
        "test" => {
            if args.len() < 3 { println!("❌ Missing image path!"); return; }
            tests::test_model(device, &args[2]);
        }
        "run" => {
            if args.len() < 3 { println!("❌ Missing image path!"); return; }
            tests::test_model(device, &args[2]);
        }
        "gui" => {
            server::start_gui(device).await;
        }
        _ => println!("❌ Unknown command: {}", args[1]),
    }
}
