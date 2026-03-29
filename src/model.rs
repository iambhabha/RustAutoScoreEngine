use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::nn::PaddingConfig2d;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: [usize; 2], device: &B::Device) -> Self {
        let config = Conv2dConfig::new([in_channels, out_channels], kernel_size)
            .with_padding(PaddingConfig2d::Same);
        let conv = config.init(device);
        let bn = BatchNormConfig::new(out_channels).init(device);
        Self { conv, bn }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        burn::tensor::activation::leaky_relu(x, 0.1)
    }
}

/// DartVision model ported from YOLOv4-tiny.
/// Input: [B, 3, 800, 800] (matching Python config: input_size=800)
/// Output grid: [B, 30, 50, 50] — 800 / 2^4 = 50
/// 30 channels = 3 anchors × 10 attrs (x, y, w, h, obj, cls0..cls4)
#[derive(Module, Debug)]
pub struct DartVisionModel<B: Backend> {
    l1: ConvBlock<B>,  // 3   -> 32
    p1: MaxPool2d,     // /2 -> 400
    l2: ConvBlock<B>,  // 32  -> 32
    p2: MaxPool2d,     // /2 -> 200
    l3: ConvBlock<B>,  // 32  -> 64
    p3: MaxPool2d,     // /2 -> 100
    l4: ConvBlock<B>,  // 64  -> 64
    p4: MaxPool2d,     // /2 ->  50
    l5: ConvBlock<B>,  // 64  -> 128
    l6: ConvBlock<B>,  // 128 -> 128
    head: Conv2d<B>,   // 128 -> 30  (detection head)
}

impl<B: Backend> DartVisionModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let l1 = ConvBlock::new(3,   32,  [3, 3], device);
        let p1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let l2 = ConvBlock::new(32,  32,  [3, 3], device);
        let p2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let l3 = ConvBlock::new(32,  64,  [3, 3], device);
        let p3 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let l4 = ConvBlock::new(64,  64,  [3, 3], device);
        let p4 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let l5 = ConvBlock::new(64,  128, [3, 3], device);
        let l6 = ConvBlock::new(128, 128, [3, 3], device);

        // 30 = 3 anchors × (x, y, w, h, obj, dart, cal1, cal2, cal3, cal4)
        let head = Conv2dConfig::new([128, 30], [1, 1]).init(device);

        Self { l1, p1, l2, p2, l3, p3, l4, p4, l5, l6, head }
    }

    /// Returns (output_50, output_50) — second is a clone kept for API compat.
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.l1.forward(x);  // [B, 32, 800, 800]
        let x = self.p1.forward(x);  // [B, 32, 400, 400]
        let x = self.l2.forward(x);  // [B, 32, 400, 400]
        let x = self.p2.forward(x);  // [B, 32, 200, 200]
        let x = self.l3.forward(x);  // [B, 64, 200, 200]
        let x = self.p3.forward(x);  // [B, 64, 100, 100]
        let x = self.l4.forward(x);  // [B, 64, 100, 100]
        let x = self.p4.forward(x);  // [B, 64,  50,  50]
        let x = self.l5.forward(x);  // [B, 128, 50,  50]
        let x = self.l6.forward(x);  // [B, 128, 50,  50]
        // NOTE: Do NOT clone here — cloning an autodiff tensor duplicates the full
        // computation graph in memory. train.rs only uses the first output.
        let out = self.head.forward(x);  // [B, 30, 50, 50]
        let out2 = out.clone().detach(); // detached copy for API compat (no grad graph)
        (out, out2)
    }
}
