use crate::data::{DartBatcher, DartDataset};
use crate::loss::diou_loss;
use crate::model::DartVisionModel;
use crate::config as cfg; // Centralized Config
use burn::data::dataset::Dataset; 
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::cast::ToElement;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::sync::Arc;
use std::fs;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct EpochState {
    current_epoch: usize,
}

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

    // 3. Create & Split Dataset
    println!("🔍 Mapping annotations from {}...", dataset_path);
    let mut dataset = DartDataset::load(dataset_path, "dataset/800");
    
    // 3.1 SHUFFLE & SPLIT (To get a valid Validation Set)
    let mut rng = thread_rng();
    dataset.annotations.shuffle(&mut rng);

    let val_count = (dataset.annotations.len() as f64 * cfg::VALIDATION_SPLIT) as usize;
    let train_count = dataset.annotations.len() - val_count;

    let (train_ann, val_ann) = dataset.annotations.split_at(train_count);
    
    let train_dataset = DartDataset {
        annotations: train_ann.to_vec(),
        base_path: dataset.base_path.clone(),
    };
    let val_dataset = DartDataset {
        annotations: val_ann.to_vec(),
        base_path: dataset.base_path.clone(),
    };

    println!("📊 Dataset split: Train={}, Val={}", train_dataset.len(), val_dataset.len());
    
    let batcher = DartBatcher::new(device.clone());

    // 4. Create DataLoaders
    println!("📦 Initializing DataLoaders (Workers: {})...", cfg::NUM_WORKERS);
    let train_dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(42)
        .num_workers(cfg::NUM_WORKERS) 
        .build(train_dataset);

    let val_dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(42)
        .num_workers(cfg::NUM_WORKERS) 
        .build(val_dataset);

    // 5. Training Loop
    println!(
        "📈 Running ULTRA Training Loop (Epochs: {})...",
        config.num_epochs
    );

    let mut current_model = model;
    
    // 5.1 Determine Start Epoch (RESUME logic)
    let epoch_file = format!("{}.epoch.json", cfg::MODEL_WEIGHTS_FILE);
    let start_epoch = if fs::metadata(&epoch_file).is_ok() {
        let content = fs::read_to_string(&epoch_file).unwrap_or_default();
        let state: EpochState = serde_json::from_str(&content).unwrap_or(EpochState { current_epoch: 1 });
        println!("🕒 Resuming from Epoch {}...", state.current_epoch);
        state.current_epoch
    } else {
        1
    };

    if start_epoch > config.num_epochs {
        println!("✅ Training already complete (Final Epoch: {} Reached).", start_epoch - 1);
        return;
    }

    for epoch in start_epoch..=config.num_epochs {
        let mut model_inner = current_model; 
        let mut batch_count = 0;
        let mut epoch_loss = 0.0;

        // --- 🎯 TRIPLE DECAY SCHEDULE ---
        let epoch_lr = if epoch <= cfg::WARMUP_EPOCHS {
            config.lr * (epoch as f64 / cfg::WARMUP_EPOCHS as f64)
        } else if epoch > (config.num_epochs * 90 / 100) {
            config.lr * 0.01 
        } else if epoch > (config.num_epochs * 75 / 100) {
            config.lr * 0.1 
        } else if epoch > (config.num_epochs * 50 / 100) {
            config.lr * 0.5 
        } else {
            config.lr
        };

        if epoch <= cfg::WARMUP_EPOCHS || epoch % 10 == 0 {
            println!("   ⚡ [Training Info] Epoch {}: Current learning rate is {:.8}", epoch, epoch_lr);
        }

        for batch in train_dataloader.iter() {
            // Forward Pass
            let (out16, _) = model_inner.forward(batch.images);

            // Calculate Loss
            let loss = diou_loss(out16, batch.targets);
            batch_count += 1;
            
            let loss_val = loss.clone().detach().into_scalar().to_f32();
            epoch_loss += loss_val;

            if batch_count % 100 == 0 || batch_count == 1 {
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

            // 5.5 Checkpoint
            if batch_count % cfg::SAVE_INTERVAL_BATCHES == 0 {
                model_inner.clone().save_file(cfg::MODEL_WEIGHTS_FILE, &recorder).ok();
            }
        }

        // --- 🏁 VALIDATION AFTER EPOCH ---
        // We clone here to ensure the training model isn't moved.
        let val_loss = validate(model_inner.clone(), &val_dataloader);
        let avg_train_loss = epoch_loss / batch_count as f32;

        println!(
            "⭐ Epoch {} results: Train Loss: {:.6} | Val Loss: {:.6}", 
            epoch, avg_train_loss, val_loss
        );

        // 6. SAVE after EACH Epoch
        model_inner
            .clone()
            .save_file(cfg::MODEL_WEIGHTS_FILE, &recorder)
            .expect("Failed to save weights");

        // 6.1 Save Epoch State
        let next_epoch = epoch + 1;
        let epoch_state = EpochState { current_epoch: next_epoch };
        if let Ok(json) = serde_json::to_string(&epoch_state) {
            fs::write(&epoch_file, json).ok();
        }

        println!("✅ Checkpoint saved: Epoch {} complete.", epoch);

        current_model = model_inner; 
    }
}

pub fn validate<B: Backend>(
    model: DartVisionModel<B>,
    dataloader: &Arc<dyn burn::data::dataloader::DataLoader<crate::data::DartBatch<B>>>
) -> f32 {
    let mut total_loss = 0.0;
    let mut count = 0;

    for batch in dataloader.iter() {
        let (out16, _) = model.forward(batch.images);
        let loss = diou_loss(out16, batch.targets);
        total_loss += loss.into_scalar().to_f32();
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }
    total_loss / count as f32
}
