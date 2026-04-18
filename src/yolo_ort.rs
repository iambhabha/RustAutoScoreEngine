use ort::session::Session;
use ort::session::builder::SessionBuilder;
use ort::value::Value;
use ndarray::Array4;
use image::{GenericImageView, imageops::FilterType};

pub struct YoloDetector {
    session: Session,
    img_size: u32,
}

impl YoloDetector {
    pub fn new(model_path: &str, img_size: u32) -> Self {
        println!("🦀 [Rust-Yolo] Loading RTX 5080 Pipeline: {}", model_path);
        let session = SessionBuilder::new().expect("Err").commit_from_file(model_path).expect("Err");
        Self { session, img_size }
    }

    pub fn detect(&mut self, image_path: &str, conf_threshold: f32) {
        let img = image::open(image_path).expect("Err");
        let resized = img.resize_exact(self.img_size, self.img_size, FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let mut input_array = Array4::<f32>::zeros((1, 3, self.img_size as usize, self.img_size as usize));
        for (x, y, pixel) in rgb.enumerate_pixels() {
            input_array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            input_array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            input_array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }

        let shape = vec![1, 3, self.img_size as i64, self.img_size as i64];
        let data_raw = input_array.into_raw_vec_and_offset().0; 
        let input_tensor = Value::from_array((shape, data_raw.into_boxed_slice())).expect("Err");
        
        let outputs = self.session.run(ort::inputs![input_tensor]).expect("Err");
        let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>().expect("Err");
        let boxes = *out_shape.get(2).unwrap() as usize;
        
        let mut darts = 0;
        let mut corners = 0;

        for i in 0..boxes {
            let mut max_conf = 0.0;
            let mut cls_id = 0;
            for c in 0..5 {
                let conf = out_data[(4 + c) * boxes + i];
                if conf > max_conf { max_conf = conf; cls_id = c; }
            }

            if max_conf > conf_threshold {
                let x = out_data[0 * boxes + i];
                let y = out_data[1 * boxes + i];
                if cls_id == 0 { 
                    darts += 1; 
                    println!("🎯 [NATIVE DART] {:.0}% @ [X:{:.0}, Y:{:.0}]", max_conf*100.0, x, y); 
                } else { 
                    corners += 1; 
                    println!("📐 [NATIVE PT {}] {:.0}% @ [X:{:.0}, Y:{:.0}]", cls_id, max_conf*100.0, x, y); 
                }
                if darts + corners > 20 { break; }
            }
        }
        println!("🚀 Rust Native Detection Finished: Found {} Darts, {} Corners", darts, corners);
    }
}
