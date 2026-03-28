use crate::model::DartVisionModel;
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use image;

pub fn run_inference<B: Backend>(device: &B::Device, image_path: &str) {
    println!("🔍 Loading model for inference...");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let model: DartVisionModel<B> = DartVisionModel::new(device);

    // Load weights
    let record = Recorder::load(&recorder, "model_weights".into(), device)
        .expect("Failed to load weights. Make sure model_weights.bin exists.");
    let model = model.load_record(record);

    println!("🖼️ Processing image: {}...", image_path);
    let img = image::open(image_path).expect("Failed to open image");
    let resized = img.resize_exact(800, 800, image::imageops::FilterType::Triangle);
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

    let data = TensorData::new(pixels, [800, 800, 3]);
    let input = Tensor::<B, 3>::from_data(data, device)
        .unsqueeze::<4>()
        .permute([0, 3, 1, 2]);

    println!("🚀 Running MODEL Prediction...");
    let (out16, _out32) = model.forward(input);

    // Post-process out16 (size [1, 30, 100, 100])
    // Decode objectness part (Channel 4 for Anchor 0)
    let obj = burn::tensor::activation::sigmoid(out16.clone().narrow(1, 4, 1));

    // Find highest confidence cell
    let (max_val, _) = obj.reshape([1, 10000]).max_dim_with_indices(1);
    let confidence: f32 = max_val
        .to_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .unwrap()[0];

    println!("--------------------------------------------------");
    println!("📊 RESULTS FOR: {}", image_path);
    println!("✨ Max Objectness: {:.2}%", confidence * 100.0);

    if confidence > 0.05 {
        println!(
            "✅ Model found something! Confidence Score: {:.4}",
            confidence
        );
    } else {
        println!("⚠️ Model confidence is too low. Training incomplete?");
    }
    println!("--------------------------------------------------");
}
