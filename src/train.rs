use crate::data::{DartBatcher, DartDataset};
use crate::loss::diou_loss;
use crate::model::DartVisionModel;
use burn::data::dataset::Dataset; // Add this trait to scope
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
    let weights_path = "model_weights.bin";
    if std::path::Path::new(weights_path).exists() {
        println!("🚀 Loading existing weights from {}...", weights_path);
        let record = Recorder::load(&recorder, "model_weights".into(), &device)
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
    println!("📦 Initializing DataLoader (Workers: 4)...");
    let dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(dataset);

    // 5. Training Loop
    println!(
        "📈 Running FULL Training Loop (Epochs: {})...",
        config.num_epochs
    );

    // Using a simple loop state for ownership safety
    let mut current_model = model;

    for epoch in 1..=config.num_epochs {
        let mut model_inner = current_model; // Move into epoch
        let mut batch_count = 0;

        for batch in dataloader.iter() {
            // Forward Pass
            let (out16, _) = model_inner.forward(batch.images);

            // Calculate Loss
            let loss = diou_loss(out16, batch.targets);
            batch_count += 1;

            // Print every 10 batches to keep terminal clean and avoid stdout sync lag
            if batch_count % 20 == 0 || batch_count == 1 {
                println!(
                    "   [Epoch {}] Batch {: >3} | Loss: {:.6}",
                    epoch,
                    batch_count,
                    loss.clone().into_scalar()
                );
            }

            // Backward & Optimization step
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model_inner);
            model_inner = optim.step(config.lr, model_inner, grads_params);

            // 5.5 Periodic Save (every 100 batches and Batch 1)
            if batch_count % 100 == 0 || batch_count == 1 {
                model_inner.clone()
                    .save_file("model_weights", &recorder)
                    .ok();
                if batch_count == 1 {
                    println!("🚀 [Checkpoint] Initial weights saved at Batch 1.");
                } else {
                    println!("🚀 [Checkpoint] Saved at Batch {}.", batch_count);
                }
            }
        }

        // 6. SAVE after EACH Epoch
        model_inner
            .clone()
            .save_file("model_weights", &recorder)
            .expect("Failed to save weights");
        println!("✅ Checkpoint saved: Epoch {} complete.", epoch);

        current_model = model_inner; // Move back out for next epoch
    }
}
