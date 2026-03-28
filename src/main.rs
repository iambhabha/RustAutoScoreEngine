use crate::model::DartVisionModel;
use crate::server::start_gui;
use crate::train::{train, TrainingConfig};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

pub mod data;
pub mod loss;
pub mod model;
pub mod scoring;
pub mod server;
pub mod train;

fn main() {
    let device = WgpuDevice::default();
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "gui" {
        println!("🌐 [Burn-DartVision] Starting Professional Dashboard...");
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(start_gui(device));
    } else if args.len() > 1 && args[1] == "test" {
        let img_path = if args.len() > 2 { &args[2] } else { "test.jpg" };
        test_model(device, img_path);
    } else {
        println!("🚀 [Burn-DartVision] Starting Full Project Training...");
        let dataset_path = "dataset/labels.json";

        let config = TrainingConfig {
            num_epochs: 10,
            batch_size: 1,
            lr: 1e-3,
        };

        train::<burn::backend::Autodiff<Wgpu>>(device, dataset_path, config);
    }
}

fn test_model(device: WgpuDevice, img_path: &str) {
    println!("🔍 Testing model on: {}", img_path);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let model = DartVisionModel::<Wgpu>::new(&device);

    let record = match recorder.load("model_weights".into(), &device) {
        Ok(r) => r,
        Err(_) => {
            println!("⚠️ Weights not found, using initial model.");
            model.clone().into_record()
        }
    };
    let model = model.load_record(record);

    let img = image::open(img_path).unwrap_or_else(|_| {
        println!("❌ Image not found at {}. Using random tensor.", img_path);
        image::DynamicImage::new_rgb8(416, 416)
    });
    let resized = img.resize_exact(416, 416, image::imageops::FilterType::Triangle);
    let pixels: Vec<f32> = resized
        .to_rgb8()
        .pixels()
        .flat_map(|p| {
            vec![
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    let tensor_data = TensorData::new(pixels, [1, 416, 416, 3]);
    let input = Tensor::<Wgpu, 4>::from_data(tensor_data, &device).permute([0, 3, 1, 2]);
    let (out, _): (Tensor<Wgpu, 4>, _) = model.forward(input);

    let obj = burn::tensor::activation::sigmoid(out.clone().narrow(1, 4, 1));
    let (max_val, _) = obj.reshape([1, 676]).max_dim_with_indices(1);

    let score = max_val
        .to_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .unwrap()[0];
    println!("📊 Max Objectness Score: {:.6}", score);
}
