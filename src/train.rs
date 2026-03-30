use crate::data::{DartBatcher, DartDataset};
use crate::loss::diou_loss;
use crate::model::DartVisionModel;
use crate::config as cfg; // Centralized Config
use burn::data::dataset::Dataset; 
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;

pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
}

pub fn train<B: AutodiffBackend>(device: Device<B>, dataset_path: &str, config: TrainingConfig) {
    // 1. Create Model
    let mut model: DartVisionModel<B> = DartVisionModel::new(&device);

    // 1.5 Load existing weights if they exist (RESUME)
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let weights_path = format!("{}.bin", cfg::MODEL_WEIGHTS_FILE);
    if std::path::Path::new(&weights_path).exists() {
        println!("🚀 Loading existing weights from {}...", weights_path);
        let record = Recorder::load(&recorder, cfg::MODEL_WEIGHTS_FILE.into(), &device)
            .expect("Failed to load weights");
        model = model.load_record(record);
    }

    // 2. Setup Optimizer
    let mut optim = AdamConfig::new().init();

    // 3. Create Dataset
    println!("🔍 Mapping annotations from {}...", dataset_path);
    let dataset = DartDataset::load(dataset_path, "dataset/800");
    println!("📊 Dataset loaded with {} examples.", dataset.len());
    let batcher = DartBatcher::new(device.clone());

    // 4. Create DataLoader
    println!("📦 Initializing DataLoader (Workers: {})...", cfg::NUM_WORKERS);
    let dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(42)
        .num_workers(cfg::NUM_WORKERS) 
        .build(dataset);

    // 5. Training Loop
    println!(
        "📈 Running ULTRA Training Loop (Epochs: {})...",
        config.num_epochs
    );

    let mut current_model = model;

    for epoch in 1..=config.num_epochs {
        let mut model_inner = current_model; 
        let mut batch_count = 0;

        // --- 🎯 TRIPLE DECAY SCHEDULE (Using cfg values for relative epochs) ---
        let epoch_lr = if epoch <= cfg::WARMUP_EPOCHS {
            config.lr * (epoch as f64 / cfg::WARMUP_EPOCHS as f64) // 1. Warmup
        } else if epoch > (config.num_epochs * 90 / 100) {
            config.lr * 0.01 // 4. Ultra Fine-tune
        } else if epoch > (config.num_epochs * 75 / 100) {
            config.lr * 0.1  // 3. Deep Fine-tune
        } else if epoch > (config.num_epochs * 50 / 100) {
            config.lr * 0.5  // 2. Mid Decay
        } else {
            config.lr        // Stable Mode
        };

        if epoch <= cfg::WARMUP_EPOCHS || epoch % 10 == 0 {
            println!("   ⚡ [Training Info] Epoch {}: Current learning rate is {:.8}", epoch, epoch_lr);
        }

        for batch in dataloader.iter() {
            // Forward Pass
            let (out16, _) = model_inner.forward(batch.images);

            // Calculate Loss
            let loss = diou_loss(out16, batch.targets);
            batch_count += 1;

            if batch_count % 100 == 0 || batch_count == 1 {
                let loss_val = loss.clone().detach().into_scalar();
                println!(
                    "   [Epoch {}] Batch {: >3} | Loss: {:.6}",
                    epoch,
                    batch_count,
                    loss_val
                );
            }

            // Backward & Optimization step
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model_inner);
            model_inner = optim.step(epoch_lr, model_inner, grads_params);

            // 5.5 Checkpoint (Using SAVE_INTERVAL_BATCHES from config)
            if batch_count % cfg::SAVE_INTERVAL_BATCHES == 0 || batch_count == 1 {
                model_inner.clone()
                    .save_file(cfg::MODEL_WEIGHTS_FILE, &recorder)
                    .ok();
            }
        }

        // 6. SAVE after EACH Epoch
        model_inner
            .clone()
            .save_file(cfg::MODEL_WEIGHTS_FILE, &recorder)
            .expect("Failed to save weights");
        println!("✅ Checkpoint saved: Epoch {} complete.", epoch);

        current_model = model_inner; 
    }
}
