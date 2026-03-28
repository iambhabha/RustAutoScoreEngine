use crate::model::DartVisionModel;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

pub fn test_model(device: WgpuDevice, img_path: &str) {
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
    
    if score > 0.1 {
        println!("✅ Model detection looks promising!");
    } else {
        println!("⚠️ Low confidence detection. Training may still be in progress.");
    }
}
