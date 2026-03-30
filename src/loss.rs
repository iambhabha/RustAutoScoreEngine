use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use crate::config as cfg; // Centralized Config

pub fn diou_loss<B: Backend>(
    bboxes_pred: Tensor<B, 4>,
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let [batch, _channels, h, w] = bboxes_pred.dims();
    let bp = bboxes_pred.reshape([batch, 3, 10, h, w]);
    let t = target.reshape([batch, 3, 10, h, w]);
    
    let eps = 1e-6;

    // 🎯 USING CENTRALIZED ACCURACY WEIGHTS FROM config.rs
    // 3. Objectness Loss (BCE)
    let obj_pred = burn::tensor::activation::sigmoid(bp.clone().narrow(2, 4, 1));
    let obj_target = t.clone().narrow(2, 4, 1);
    
    let pos_loss = obj_target.clone().mul(obj_pred.clone().add_scalar(eps).log()).neg();
    let neg_loss = obj_target.clone().neg().add_scalar(1.0)
        .mul(obj_pred.clone().neg().add_scalar(1.0 + eps).log()).neg();
    
    // Weight positive samples heavily (sparsity)
    let obj_loss = pos_loss.mul_scalar(cfg::WEIGHT_OBJ_LOSS).add(neg_loss).mean();

    // 4. Class Loss (Full BCE for all 5 channels)
    // bp channels 5-9: Dart, Cal1, Cal2, Cal3, Cal4
    let cls_pred = burn::tensor::activation::sigmoid(bp.clone().narrow(2, 5, 5));
    let cls_target = t.clone().narrow(2, 5, 5);
    
    let cls_pos_loss = cls_target.clone().mul(cls_pred.clone().add_scalar(eps).log()).neg();
    let cls_neg_loss = cls_target.clone().neg().add_scalar(1.0)
        .mul(cls_pred.clone().neg().add_scalar(1.0 + eps).log()).neg();
    
    let class_loss = cls_pos_loss.add(cls_neg_loss)
        .mul(obj_target.clone()) // Mask class loss where there is no object
        .mean()
        .mul_scalar(cfg::WEIGHT_CLASS_LOSS);

    // 📐 PRECISION COORDINATE LOSS (MSE)
    let b_xy_pred = burn::tensor::activation::sigmoid(bp.clone().narrow(2, 0, 2));
    let b_xy_target = t.clone().narrow(2, 0, 2);
    let xy_loss = b_xy_pred.sub(b_xy_target).powf_scalar(2.0)
        .mul(obj_target.clone())
        .mean()
        .mul_scalar(cfg::WEIGHT_XY_LOSS); 

    let b_wh_pred = burn::tensor::activation::sigmoid(bp.clone().narrow(2, 2, 2));
    let b_wh_target = t.clone().narrow(2, 2, 2);
    let wh_loss = b_wh_pred.sub(b_wh_target).powf_scalar(2.0)
        .mul(obj_target)
        .mean()
        .mul_scalar(cfg::WEIGHT_WH_LOSS);

    obj_loss.add(class_loss).add(xy_loss).add(wh_loss)
}
