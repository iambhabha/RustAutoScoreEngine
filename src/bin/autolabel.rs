use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use ndarray::{Axis, ArrayView, Ix2};
use image::{imageops::FilterType, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
struct Annotation {
    img_folder: String,
    img_name: String,
    bbox: [u32; 4],
    xy: Vec<Vec<f32>>,
}

fn draw_rect(img: &mut RgbImage, cx: f32, cy: f32, bw: f32, bh: f32, color: Rgb<u8>) {
    let (w, h) = img.dimensions();
    let x1 = ((cx - bw / 2.0) * w as f32) as i32;
    let y1 = ((cy - bh / 2.0) * h as f32) as i32;
    let x2 = ((cx + bw / 2.0) * w as f32) as i32;
    let y2 = ((cy + bh / 2.0) * h as f32) as i32;
    
    // Draw edges
    for x in x1..=x2 {
        for t in 0..2 {
            if x >= 0 && x < w as i32 {
                if y1 + t >= 0 && y1 + t < h as i32 { img.put_pixel(x as u32, (y1 + t) as u32, color); }
                if y2 - t >= 0 && y2 - t < h as i32 { img.put_pixel(x as u32, (y2 - t) as u32, color); }
            }
        }
    }
    for y in y1..=y2 {
        for t in 0..2 {
            if y >= 0 && y < h as i32 {
                if x1 + t >= 0 && x1 + t < w as i32 { img.put_pixel((x1 + t) as u32, y as u32, color); }
                if x2 - t >= 0 && x2 - t < w as i32 { img.put_pixel((x2 - t) as u32, y as u32, color); }
            }
        }
    }
}

// Simple Pixel Art Text Renderer (3x5 pixels per char)
fn draw_char(img: &mut RgbImage, px: i32, py: i32, c: char, color: Rgb<u8>) {
    let (w, h) = img.dimensions();
    let pixels = match c {
        'C' => vec![(1,0),(2,0),(0,1),(0,2),(0,3),(1,4),(2,4)],
        'D' => vec![(0,0),(1,0),(0,1),(2,1),(0,2),(2,2),(0,3),(2,3),(0,4),(1,4)],
        '1' => vec![(1,0),(1,1),(1,2),(1,3),(1,4)],
        '2' => vec![(0,0),(1,0),(2,0),(2,1),(0,2),(1,2),(2,2),(0,3),(0,4),(1,4),(2,4)],
        '3' => vec![(0,0),(1,0),(2,0),(2,1),(1,2),(2,3),(0,4),(1,4),(2,4)],
        '4' => vec![(0,0),(2,0),(0,1),(2,1),(0,2),(1,2),(2,2),(2,3),(2,4)],
        _ => vec![],
    };
    for (dx, dy) in pixels {
        for rx in 0..2 {
            for ry in 0..2 {
                let nx = px + (dx * 2) + rx;
                let ny = py + (dy * 2) + ry;
                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                    img.put_pixel(nx as u32, ny as u32, color);
                }
            }
        }
    }
}

fn draw_label(img: &mut RgbImage, cx: f32, cy: f32, bh: f32, text: &str, color: Rgb<u8>) {
    let (w, h) = img.dimensions();
    let px = (cx * w as f32) as i32 - 10;
    let py = ((cy - bh / 2.0) * h as f32) as i32 - 12;
    
    let mut curr_x = px;
    for c in text.chars() {
        draw_char(img, curr_x, py, c, color);
        curr_x += 8;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Pure Rust Auto-Labeler with Visuals...");
    
    let viz_root = Path::new("dataset/labeled_rust_viz");
    fs::create_dir_all(viz_root)?;

    let model_path = "model.onnx";
    if !Path::new(model_path).exists() {
        return Err("❌ model.onnx not found.".into());
    }
    
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    println!("✅ Model loaded.");

    let dataset_dir = "dataset/800";
    let mut auto_labels = HashMap::new();
    let mut global_idx = 0;

    println!("🎨 Color Guide for Visuals:");
    println!("   🔴 Red    = Corner 1");
    println!("   🟢 Green  = Corner 2");
    println!("   🔵 Blue   = Corner 3");
    println!("   🟡 Yellow = Corner 4");
    println!("   ⚪ White  = Darts");
    println!("---------------------------");

    let entries = fs::read_dir(dataset_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect::<Vec<_>>();

    for folder_entry in entries {
        let folder_name = folder_entry.file_name().into_string().unwrap();
        let folder_viz_path = viz_root.join(&folder_name);
        fs::create_dir_all(&folder_viz_path)?;

        let images = fs::read_dir(folder_entry.path())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "JPG" || ext == "jpg" || ext == "png"))
            .collect::<Vec<_>>();

        println!("📂 Processing: {} ({} images)", folder_name, images.len());

        for img_entry in images {
            let img_path = img_entry.path();
            let img_name = img_entry.file_name().into_string().unwrap();

            let img = image::open(&img_path)?;
            let mut viz_img = img.to_rgb8();
            let resized = img.resize_exact(640, 640, FilterType::Triangle).to_rgb8();
            
            let mut input_data = vec![0.0f32; 3 * 640 * 640];
            for (x, y, pixel) in resized.enumerate_pixels() {
                input_data[0 * 640 * 640 + (y as usize) * 640 + (x as usize)] = pixel[0] as f32 / 255.0;
                input_data[1 * 640 * 640 + (y as usize) * 640 + (x as usize)] = pixel[1] as f32 / 255.0;
                input_data[2 * 640 * 640 + (y as usize) * 640 + (x as usize)] = pixel[2] as f32 / 255.0;
            }

            let input_tensor = Value::from_array(([1, 3, 640, 640], input_data.into_boxed_slice()))?;
            let outputs = session.run(ort::inputs![input_tensor])?;
            let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
            
            let (shape, data) = output_tensor;
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let boxes_array = ArrayView::from_shape(dims, data)?
                .to_owned()
                .remove_axis(Axis(0))
                .into_dimensionality::<Ix2>()?; 
            
            let mut top_corners: Vec<Option<(f32, Vec<f32>, f32, f32)>> = vec![None; 4];
            let mut darts: Vec<(f32, Vec<f32>, f32, f32)> = Vec::new();

            for i in 0..8400 {
                let col = boxes_array.column(i);
                for cls in 0..5 {
                    let score = col[4 + cls];
                    if score > 0.4 {
                        let cx = col[0] / 640.0;
                        let cy = col[1] / 640.0;
                        let bw = col[2] / 640.0;
                        let bh = col[3] / 640.0;
                        if cls == 0 {
                            darts.push((score, vec![cx, cy], bw, bh));
                        } else {
                            let c_idx = cls as usize - 1;
                            if top_corners[c_idx].is_none() || score > top_corners[c_idx].as_ref().unwrap().0 {
                                top_corners[c_idx] = Some((score, vec![cx, cy], bw, bh));
                            }
                        }
                    }
                }
            }

            // Build final coordinates and DRAW them
            let mut final_xy = Vec::new();
            let colors = [
                Rgb([255, 0, 0]),     // C1: Red
                Rgb([0, 255, 0]),     // C2: Green
                Rgb([0, 0, 255]),     // C3: Blue
                Rgb([255, 255, 0]),   // C4: Yellow
            ];

            for (i, opt) in top_corners.iter().enumerate() {
                if let Some((_, p, _bw, _bh)) = opt {
                    final_xy.push(p.clone());
                    // Fixed small box (12px on 640px = 12/640)
                    let sz = 12.0 / 640.0; 
                    draw_rect(&mut viz_img, p[0], p[1], sz, sz, colors[i]);
                    draw_label(&mut viz_img, p[0], p[1], sz, &format!("C{}", i + 1), colors[i]);
                } else {
                    final_xy.push(vec![0.0, 0.0]);
                }
            }
            
            darts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for (_, p, _bw, _bh) in darts.iter().take(5) {
                final_xy.push(p.clone());
                let sz = 12.0 / 640.0;
                draw_rect(&mut viz_img, p[0], p[1], sz, sz, Rgb([255, 255, 255])); 
                draw_label(&mut viz_img, p[0], p[1], sz, "D", Rgb([255, 255, 255]));
            }

            // Save viz image
            viz_img.save(folder_viz_path.join(&img_name))?;

            auto_labels.insert(global_idx.to_string(), Annotation {
                img_folder: folder_name.clone(),
                img_name,
                bbox: [0, 800, 0, 800],
                xy: final_xy,
            });
            global_idx += 1;
        }
    }

    let json_data = serde_json::to_string_pretty(&auto_labels)?;
    fs::write("dataset/labels_auto_rust.json", json_data)?;
    println!("✅ DONE! Visuals saved in 'dataset/labeled_rust_viz/'");

    Ok(())
}
