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

#[derive(Module, Debug)]
pub struct DartVisionModel<B: Backend> {
    // Lean architecture: High resolution (800x800) but low channel count to fix GPU OOM
    l1: ConvBlock<B>, // 3 -> 16
    p1: MaxPool2d,
    l2: ConvBlock<B>, // 16 -> 16
    p2: MaxPool2d,
    l3: ConvBlock<B>, // 16 -> 32
    p3: MaxPool2d,
    l4: ConvBlock<B>, // 32 -> 32
    p4: MaxPool2d,
    l5: ConvBlock<B>, // 32 -> 64
    l6: ConvBlock<B>, // 64 -> 64

    head_32: Conv2d<B>, // Final detection head
}

impl<B: Backend> DartVisionModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let l1 = ConvBlock::new(3, 16, [3, 3], device);
        let p1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        
        let l2 = ConvBlock::new(16, 16, [3, 3], device);
        let p2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        
        let l3 = ConvBlock::new(16, 32, [3, 3], device);
        let p3 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        
        let l4 = ConvBlock::new(32, 32, [3, 3], device);
        let p4 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        
        let l5 = ConvBlock::new(32, 64, [3, 3], device);
        let l6 = ConvBlock::new(64, 64, [3, 3], device);

        // 30 channels = 3 anchors * (x,y,w,h,obj,p0...p4)
        let head_32 = Conv2dConfig::new([64, 30], [1, 1]).init(device);

        Self { l1, p1, l2, p2, l3, p3, l4, p4, l5, l6, head_32 }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.l1.forward(x); // 800
        let x = self.p1.forward(x); // 400
        let x = self.l2.forward(x); // 400
        let x = self.p2.forward(x); // 200
        let x = self.l3.forward(x); // 200
        let x = self.p3.forward(x); // 100
        let x = self.l4.forward(x); // 100
        let x = self.p4.forward(x); // 50
        
        let x50 = self.l5.forward(x); // 50
        let x50 = self.l6.forward(x50); // 50

        let out50 = self.head_32.forward(x50);
        
        (out50.clone(), out50)
    }
}
