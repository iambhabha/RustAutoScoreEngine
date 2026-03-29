use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Annotation {
    pub img_folder: String,
    pub img_name: String,
    pub bbox: Vec<i32>,
    pub xy: Vec<Vec<f32>>,
}

pub struct DartDataset {
    pub annotations: Vec<Annotation>,
    pub base_path: String,
}

impl DartDataset {
    pub fn load(json_path: &str, base_path: &str) -> Self {
        let file = File::open(json_path).expect("Labels JSON not found");
        let reader = BufReader::new(file);
        let raw_data: HashMap<String, Annotation> = serde_json::from_reader(reader).expect("JSON parse error");
        
        let mut annotations: Vec<Annotation> = raw_data.into_values().collect();
        annotations.sort_by(|a, b| a.img_name.cmp(&b.img_name));

        Self {
            annotations,
            base_path: base_path.to_string(),
        }
    }
}

impl burn::data::dataset::Dataset<Annotation> for DartDataset {
    fn get(&self, index: usize) -> Option<Annotation> {
        self.annotations.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.annotations.len()
    }
}

#[derive(Clone, Debug)]
pub struct DartBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

#[derive(Clone, Debug)]
pub struct DartBatcher<B: Backend> {
    device: Device<B>,
}

use burn::data::dataloader::batcher::Batcher;

impl<B: Backend> Batcher<Annotation, DartBatch<B>> for DartBatcher<B> {
    fn batch(&self, items: Vec<Annotation>) -> DartBatch<B> {
        self.batch_manual(items)
    }
}

impl<B: Backend> DartBatcher<B> {
    pub fn new(device: Device<B>) -> Self {
        Self { device }
    }

    pub fn batch_manual(&self, items: Vec<Annotation>) -> DartBatch<B> {
        let batch_size = items.len();
        // Use 800 to match original Python training config (configs/deepdarts_d1.yaml: input_size: 800)
        let input_res: usize = 800;
        // For tiny YOLO: grid = input_res / 16. 800/16 = 50
        let grid_size: usize = 50;
        let num_anchors: usize = 3;
        let num_attrs: usize = 10; // x, y, w, h, obj, cls0..cls4
        let num_channels: usize = num_anchors * num_attrs; // = 30

        let mut images_list = Vec::with_capacity(batch_size);
        let mut target_raw = vec![0.0f32; batch_size * num_channels * grid_size * grid_size];

        for (b_idx, item) in items.iter().enumerate() {
            // 1. Process Image
            let path = format!("dataset/800/{}/{}", item.img_folder, item.img_name);
            let img = image::open(&path).unwrap_or_else(|_| {
                println!("⚠️ [Data] Image not found: {}", path);
                image::DynamicImage::new_rgb8(input_res as u32, input_res as u32)
            });
            let resized = img.resize_exact(input_res as u32, input_res as u32, image::imageops::FilterType::Triangle);
            let pixels: Vec<f32> = resized.to_rgb8().pixels()
                .flat_map(|p| vec![p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
                .collect();
            images_list.push(TensorData::new(pixels, [input_res, input_res, 3]));

            for (i, p) in item.xy.iter().enumerate() {
                // Clamp coordinates to valid grid range
                let norm_x = p[0].clamp(0.0, 1.0 - 1e-5);
                let norm_y = p[1].clamp(0.0, 1.0 - 1e-5);

                let gx = (norm_x * grid_size as f32).floor() as usize;
                let gy = (norm_y * grid_size as f32).floor() as usize;

                // Grid-relative offset (0..1 within cell)
                let tx = norm_x * grid_size as f32 - gx as f32;
                let ty = norm_y * grid_size as f32 - gy as f32;

                // Python convention: cal points i=0..3 -> cls=1..4, dart i>=4 -> cls=0
                let cls = if i < 4 { i + 1 } else { 0 };

                // Assign this keypoint to anchor (cls % num_anchors) so all 3 anchors get used
                let anchor_idx = cls % num_anchors;

                // Flat index layout: [batch, anchor, attr, gy, gx]
                // => flat = b * (3*10*G*G) + anchor * (10*G*G) + attr * (G*G) + gy*G + gx
                let cell_base = b_idx * num_channels * grid_size * grid_size
                    + anchor_idx * num_attrs * grid_size * grid_size
                    + gy * grid_size
                    + gx;

                target_raw[cell_base + 0 * grid_size * grid_size] = tx;   // x offset
                target_raw[cell_base + 1 * grid_size * grid_size] = ty;   // y offset
                target_raw[cell_base + 2 * grid_size * grid_size] = 0.025; // w (bbox_size from config)
                target_raw[cell_base + 3 * grid_size * grid_size] = 0.025; // h
                target_raw[cell_base + 4 * grid_size * grid_size] = 1.0;   // objectness
                target_raw[cell_base + (5 + cls) * grid_size * grid_size] = 1.0; // class prob
            }
        }

        let images = Tensor::stack(
            images_list.into_iter().map(|d| Tensor::<B, 3>::from_data(d, &self.device)).collect(),
            0
        ).permute([0, 3, 1, 2]);

        let targets = Tensor::from_data(
            TensorData::new(target_raw, [batch_size, num_channels, grid_size, grid_size]),
            &self.device
        );

        DartBatch { images, targets }
    }
}
