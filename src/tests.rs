use crate::yolo_ort::YoloDetector;

pub fn test_model(_device: burn::backend::wgpu::WgpuDevice, img_path: &str) {
    println!("🔍 [Native-Rust] Testing YOLOv8 Model on: {}", img_path);
    
    // Initialize our new Yolo Ort Detector (800x800)
    let model_path = "best.onnx";
    let mut detector = YoloDetector::new(model_path, 800);

    // Run detection with same 1:1 logic
    println!("🚀 Starting Real-time RTX 5080 Inference...");
    detector.detect(img_path, 0.50);

    println!("-------------------------------------------");
    println!("✅ Rust Detection Complete!");
    println!("💡 Use 'cargo run -- test <image_path>' to repeat.");
}
