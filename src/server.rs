use axum::{
    extract::{DefaultBodyLimit, Multipart},
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use crate::yolo_ort::YoloDetector;
use serde_json::json;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use axum_server::tls_rustls::RustlsConfig;
use std::path::PathBuf;
use tower_http::cors::CorsLayer;

async fn ensure_certs() -> RustlsConfig {
    let cert_path = PathBuf::from("cert.pem");
    let key_path = PathBuf::from("key.pem");
    if !cert_path.exists() || !key_path.exists() {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string(), "127.0.0.1".into()]).unwrap();
        std::fs::write(&cert_path, cert.cert.pem()).unwrap();
        std::fs::write(&key_path, cert.key_pair.serialize_pem()).unwrap();
    }
    RustlsConfig::from_pem_file(cert_path, key_path).await.unwrap()
}

pub async fn start_gui(_device: burn::backend::wgpu::WgpuDevice) {
    let port = 8080;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let config = ensure_certs().await;

    println!("🚀 [DartVision-Rust] RTX 5080 Engine starting on HTTPS -> https://localhost:{}", port);
    
    // Initialize our NEW YOLO Detector
    // Wrapped in Arc<Mutex> so it can be shared safely between requests
    let detector = Arc::new(Mutex::new(YoloDetector::new("best.onnx", 800)));

    let app = Router::new()
        .route("/", get(|| async { Html(include_str!("../static/index.html")) }))
        .route("/api/predict", post(predict_handler))
        .with_state(detector)
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(CorsLayer::permissive());

    axum_server::bind_rustls(addr, config)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn predict_handler(
    axum::extract::State(detector): axum::extract::State<Arc<Mutex<YoloDetector>>>,
    mut multipart: Multipart,
) -> Json<serde_json::Value> {
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("image") {
            let bytes = field.bytes().await.unwrap().to_vec();
            
            // Save temporary image for detector (standard logic)
            let temp_path = "temp_frame.jpg";
            let _ = std::fs::write(temp_path, &bytes);

            // Run Native YOLOv8 Detection
            println!("🔍 [API] Running RTX 5080 Inference...");
            let mut det = detector.lock().unwrap();
            det.detect(temp_path, 0.40); // 40% threshold for real-time

            // Return success to GUI
            return Json(json!({
                "status": "success",
                "message": "Detection processed in Rust (Check Terminal for Coordinates)",
                "engine": "YOLOv8-ORT-RTX5080"
            }));
        }
    }
    Json(json!({"status": "error"}))
}
