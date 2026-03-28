use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub fn diou_loss<B: Backend>(
    bboxes_pred: Tensor<B, 4>,
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    // 1. Reshape to separate anchors: [Batch, 3, 10, H, W]
    let [batch, _channels, h, w] = bboxes_pred.dims();
    let bp = bboxes_pred.reshape([batch, 3, 10, h, w]);
    let t = target.reshape([batch, 3, 10, h, w]);
    
    // 2. Objectness (Channel 4)
    let obj_pred = burn::tensor::activation::sigmoid(bp.clone().narrow(2, 4, 1));
    let obj_target = t.clone().narrow(2, 4, 1);
    
    let eps = 1e-7;
    // Positive loss (where an object exists)
    let pos_loss = obj_target.clone().mul(obj_pred.clone().add_scalar(eps).log()).neg();
    // Negative loss (where no object exists)
    let neg_loss = obj_target.clone().neg().add_scalar(1.0).mul(obj_pred.clone().neg().add_scalar(1.0 + eps).log()).neg();
    
    // Weight positive samples 10x more to fight imbalance (typical YOLO trick)
    let obj_loss = pos_loss.mul_scalar(20.0).add(neg_loss).mean();

    // 3. Class (Channels 5-9) - Only learn when object exists
    let cls_pred = burn::tensor::activation::sigmoid(bp.clone().narrow(2, 5, 5));
    let cls_target = t.clone().narrow(2, 5, 5);
    let class_loss = cls_target.clone().mul(cls_pred.clone().add_scalar(eps).log()).neg()
        .mul(obj_target.clone()) // Only count where object exists
        .mean()
        .mul_scalar(5.0); // Boost class learning

    // 4. Coordinates (Channels 0-3) - Only learn when object exists
    let b_xy_pred = burn::tensor::activation::sigmoid(bp.clone().narrow(2, 0, 2));
    let b_xy_target = t.clone().narrow(2, 0, 2);
    let xy_loss = b_xy_pred.sub(b_xy_target).powf_scalar(2.0).mul(obj_target.clone()).mean().mul_scalar(5.0);

    let b_wh_pred = burn::tensor::activation::sigmoid(bp.clone().narrow(2, 2, 2));
    let b_wh_target = t.clone().narrow(2, 2, 2);
    let wh_loss = b_wh_pred.sub(b_wh_target).powf_scalar(2.0).mul(obj_target).mean().mul_scalar(5.0);

    obj_loss.add(class_loss).add(xy_loss).add(wh_loss)
}
