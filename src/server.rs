use crate::model::DartVisionModel;
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tower_http::cors::CorsLayer;

#[derive(Debug)]
struct PredictResult {
    confidence: f32,
    keypoints: Vec<f32>,
    scores: Vec<String>,
}

struct PredictRequest {
    image_bytes: Vec<u8>,
    response_tx: oneshot::Sender<PredictResult>,
}

pub async fn start_gui(device: WgpuDevice) {
    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("🚀 [DartVision-GUI] Starting on http://127.0.0.1:8080",);

    let (tx, mut rx) = mpsc::channel::<PredictRequest>(10);

    let worker_device = device.clone();
    std::thread::spawn(move || {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let model = DartVisionModel::<Wgpu>::new(&worker_device);
        let record = match recorder.load("model_weights".into(), &worker_device) {
            Ok(r) => r,
            Err(_) => {
                println!("⚠️ [DartVision] No 'model_weights.bin' yet. Using initial weights...");
                model.clone().into_record()
            }
        };
        let model = model.load_record(record);

        while let Some(req) = rx.blocking_recv() {
            let start_time = std::time::Instant::now();
            
            let img = image::load_from_memory(&req.image_bytes).unwrap();
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
            let input =
                Tensor::<Wgpu, 4>::from_data(tensor_data, &worker_device).permute([0, 3, 1, 2]);

            let (out16, _) = model.forward(input);

            // 1. Reshape to separate anchors: [1, 3, 10, 26, 26]
            let out_reshaped = out16.reshape([1, 3, 10, 26, 26]);

            // 1.5 Debug: Raw Statistics
            println!(
                "🔍 [Model Stats] Raw Min: {:.4}, Max: {:.4}",
                out_reshaped.clone().min().into_scalar(),
                out_reshaped.clone().max().into_scalar()
            );

            let mut final_points = vec![0.0f32; 8]; // 4 corners
            let mut max_conf = 0.0f32;

            // 2. Extract best calibration corner for each class 1 to 4
            for cls_idx in 1..=4 {
                let mut best_s = -1.0f32;
                let mut best_pt = [0.0f32; 2];
                let mut best_anchor = 0;
                let mut best_grid = (0, 0);

                for anchor in 0..3 {
                    let obj = burn::tensor::activation::sigmoid(
                        out_reshaped.clone().narrow(1, anchor, 1).narrow(2, 4, 1),
                    );
                    let prob = burn::tensor::activation::sigmoid(
                        out_reshaped
                            .clone()
                            .narrow(1, anchor, 1)
                            .narrow(2, 5 + cls_idx, 1),
                    );
                    let score = obj.mul(prob);

                    let (val, idx) = score.reshape([1, 676]).max_dim_with_indices(1);
                    let s = val.to_data().convert::<f32>().as_slice::<f32>().unwrap()[0];
                    if s > best_s {
                        best_s = s;
                        best_anchor = anchor;
                        let f_idx =
                            idx.to_data().convert::<i32>().as_slice::<i32>().unwrap()[0] as usize;
                        best_grid = (f_idx % 26, f_idx / 26);

                        let sx = burn::tensor::activation::sigmoid(
                            out_reshaped
                                .clone()
                                .narrow(1, anchor, 1)
                                .narrow(2, 0, 1)
                                .slice([
                                    0..1,
                                    0..1,
                                    0..1,
                                    best_grid.1..best_grid.1 + 1,
                                    best_grid.0..best_grid.0 + 1,
                                ]),
                        )
                        .to_data()
                        .convert::<f32>()
                        .as_slice::<f32>()
                        .unwrap()[0];
                        let sy = burn::tensor::activation::sigmoid(
                            out_reshaped
                                .clone()
                                .narrow(1, anchor, 1)
                                .narrow(2, 1, 1)
                                .slice([
                                    0..1,
                                    0..1,
                                    0..1,
                                    best_grid.1..best_grid.1 + 1,
                                    best_grid.0..best_grid.0 + 1,
                                ]),
                        )
                        .to_data()
                        .convert::<f32>()
                        .as_slice::<f32>()
                        .unwrap()[0];
                        
                        // Reconstruct Absolute Normalized Coord (0-1)
                        best_pt = [
                            (best_grid.0 as f32 + sx) / 26.0,
                            (best_grid.1 as f32 + sy) / 26.0,
                        ];
                    }
                }

                final_points[(cls_idx - 1) * 2] = best_pt[0];
                final_points[(cls_idx - 1) * 2 + 1] = best_pt[1];
                if best_s > max_conf {
                    max_conf = best_s;
                }
                println!(
                    "   [Debug Cal{}] Anchor: {}, Conf: {:.4}, Cell: {:?}, Coord: [{:.3}, {:.3}]",
                    cls_idx, best_anchor, best_s, best_grid, best_pt[0], best_pt[1]
                );
            }

            // 3. Calibration Estimation (Python logic: est_cal_pts)
            // If one calibration point is missing, estimate it using symmetry
            let mut valid_cal_count = 0;
            let mut missing_idx = -1;
            for i in 0..4 {
                if final_points[i*2] > 0.01 || final_points[i*2+1] > 0.01 {
                    valid_cal_count += 1;
                } else {
                    missing_idx = i as i32;
                }
            }

            if valid_cal_count == 3 {
                println!("⚠️ [Calibration Recovery] Estimating missing point Cal{}...", missing_idx + 1);
                match missing_idx {
                    0 | 1 => { // Top points missing, use bottom points center
                        let cx = (final_points[4] + final_points[6]) / 2.0;
                        let cy = (final_points[5] + final_points[7]) / 2.0;
                        if missing_idx == 0 {
                            final_points[0] = 2.0 * cx - final_points[2];
                            final_points[1] = 2.0 * cy - final_points[3];
                        } else {
                            final_points[2] = 2.0 * cx - final_points[0];
                            final_points[3] = 2.0 * cy - final_points[1];
                        }
                    },
                    2 | 3 => { // Bottom points missing, use top points center
                        let cx = (final_points[0] + final_points[2]) / 2.0;
                        let cy = (final_points[1] + final_points[3]) / 2.0;
                        if missing_idx == 2 {
                            final_points[4] = 2.0 * cx - final_points[6];
                            final_points[5] = 2.0 * cy - final_points[7];
                        } else {
                            final_points[6] = 2.0 * cx - final_points[4];
                            final_points[7] = 2.0 * cy - final_points[5];
                        }
                    },
                    _ => {}
                }
            }

            // 4. Extract best dart (Class 0) - Find candidates across all anchors
            println!("   [Debug Dart] Searching for Candidates...");
            let mut dart_candidates = vec![];
            for anchor in 0..3 {
                let obj = burn::tensor::activation::sigmoid(
                    out_reshaped.clone().narrow(1, anchor, 1).narrow(2, 4, 1),
                );
                let prob = burn::tensor::activation::sigmoid(
                    out_reshaped.clone().narrow(1, anchor, 1).narrow(2, 5, 1),
                );
                let score = obj.mul(prob).reshape([1, 676]);

                let (val, idx) = score.max_dim_with_indices(1);
                let s = val.to_data().convert::<f32>().as_slice::<f32>().unwrap()[0];
                let f_idx = idx.to_data().convert::<i32>().as_slice::<i32>().unwrap()[0] as usize;

                let gx = f_idx % 26;
                let gy = f_idx / 26;

                let dsx = burn::tensor::activation::sigmoid(
                    out_reshaped
                        .clone()
                        .narrow(1, anchor, 1)
                        .narrow(2, 0, 1)
                        .slice([0..1, 0..1, 0..1, gy..gy + 1, gx..gx + 1]),
                )
                .to_data()
                .convert::<f32>()
                .as_slice::<f32>()
                .unwrap()[0];
                let dsy = burn::tensor::activation::sigmoid(
                    out_reshaped
                        .clone()
                        .narrow(1, anchor, 1)
                        .narrow(2, 1, 1)
                        .slice([0..1, 0..1, 0..1, gy..gy + 1, gx..gx + 1]),
                )
                .to_data()
                .convert::<f32>()
                .as_slice::<f32>()
                .unwrap()[0];

                let dx = (gx as f32 + dsx) / 26.0;
                let dy = (gy as f32 + dsy) / 26.0;

                if s > 0.005 {
                    println!(
                        "     - A{} Best Cell: ({},{}), Conf: {:.4}, Coord: [{:.3}, {:.3}]",
                        anchor, gx, gy, s, dx, dy
                    );
                    dart_candidates.push((s, [dx, dy]));
                }
            }

            // Pick the best dart candidate among all anchors
            dart_candidates
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            if let Some((s, pt)) = dart_candidates.first() {
                if *s > 0.05 {
                    final_points.push(pt[0]);
                    final_points.push(pt[1]);
                    println!(
                        "   ✅ Best Dart Picked: Conf: {:.2}%, Coord: {:?}",
                        s * 100.0,
                        pt
                    );
                }
            }

            let mut final_scores = vec![];

            // Calculate scores if we have calibration points and at least one dart
            if final_points.len() >= 10 {
                use crate::scoring::{calculate_dart_score, ScoringConfig};
                let config = ScoringConfig::default();
                let cal_pts = [
                    [final_points[0], final_points[1]],
                    [final_points[2], final_points[3]],
                    [final_points[4], final_points[5]],
                    [final_points[6], final_points[7]],
                ];

                for dart_chunk in final_points[8..].chunks(2) {
                    if dart_chunk.len() == 2 {
                        let dart_pt = [dart_chunk[0], dart_chunk[1]];
                        let (s_val, label) = calculate_dart_score(&cal_pts, &dart_pt, &config);
                        final_scores.push(label.clone());
                        println!("   [Debug Score] Label: {} (Val: {})", label, s_val);
                    }
                }
            }

            let duration = start_time.elapsed();
            println!("⚡ [Inference Performance] Total Latency: {:.2?}", duration);

            println!("🎯 [Final Result] Top Confidence: {:.2}%", max_conf * 100.0);
            let class_names = ["Cal1", "Cal2", "Cal3", "Cal4", "Dart"];
            for (i, pts) in final_points.chunks(2).enumerate() {
                let name = class_names.get(i).unwrap_or(&"Dart");
                let label = final_scores
                    .get(i.saturating_sub(4))
                    .cloned()
                    .unwrap_or_default();
                println!(
                    "   - {}: [x: {:.3}, y: {:.3}] {}",
                    name, pts[0], pts[1], label
                );
            }

            let _ = req.response_tx.send(PredictResult {
                confidence: max_conf,
                keypoints: final_points,
                scores: final_scores,
            });
        }
    });

    let state = Arc::new(tx);

    let app = Router::new()
        .route(
            "/",
            get(|| async { Html(include_str!("../static/index.html")) }),
        )
        .route("/api/predict", post(predict_handler))
        .with_state(state)
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn predict_handler(
    State(tx): State<Arc<mpsc::Sender<PredictRequest>>>,
    mut multipart: Multipart,
) -> Json<serde_json::Value> {
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("image") {
            let bytes = match field.bytes().await {
                Ok(b) => b.to_vec(),
                Err(_) => continue,
            };
            let (res_tx, res_rx) = oneshot::channel();
            let _ = tx
                .send(PredictRequest {
                    image_bytes: bytes,
                    response_tx: res_tx,
                })
                .await;
            let result = res_rx.await.unwrap_or(PredictResult {
                confidence: 0.0,
                keypoints: vec![],
                scores: vec![],
            });

            return Json(json!({
                "status": "success",
                "confidence": result.confidence,
                "keypoints": result.keypoints,
                "scores": result.scores,
                "message": if result.confidence > 0.1 {
                    format!("✅ Found {} darts! High confidence: {:.1}%", result.scores.len(), result.confidence * 100.0)
                } else {
                    "⚠️ Low confidence detection - no dart score could be verified.".to_string()
                }
            }));
        }
    }
    Json(json!({"status": "error", "message": "No image field found"}))
}
