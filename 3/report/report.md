# Instance Segmentation of Cells using Deep Learning

- **Name:** 馬楷翔
- **Student ID:** 110550074

## 1. Introduction

This report details the development and evaluation of an instance segmentation model for identifying and delineating individual cells in microscopy images. Instance segmentation is a challenging computer vision task that requires not only detecting objects but also predicting a pixel-level mask for each distinct instance. The core approach involves leveraging state-of-the-art deep learning architectures, fine-tuned on a custom dataset of cell images.

The primary model investigated and submitted is a **Cascade Mask R-CNN with a ResNeXt-152 backbone** (referred to as `grcnn`), which demonstrated strong performance. As an additional experiment, a **PointRend R-CNN X-101-FPN** model was also trained and evaluated to explore its potential for refining mask quality. This report will present the methodologies for both approaches, compare their results, and discuss the findings. Ultimately, the Cascade Mask R-CNN model achieved a higher Average Precision (AP) of 0.41 compared to the PointRend model's 0.37 AP on our custom cell dataset.

The project explores the efficacy of the powerful Cascade Mask R-CNN model and further discusses the PointRend experiment, along with potential future improvements like SoftNMS for handling overlapping instances and model ensembling using Weighted Boxes Fusion (WBF).

## 2. Method

### 2.1. Dataset and Preprocessing

- **Dataset**: The experiments were conducted on a custom "cells" dataset, split into `cells_train` (167 images) and `cells_val` (42 images). The dataset contains 4 classes of cells.
  - Training set class distribution:
    - class1: 11,591 instances
    - class2: 11,162 instances
    - class3: 521 instances
    - class4: 529 instances
    - Total: 23,803 instances
  - Validation set class distribution:
    _class1: 2,946 instances
    _ class2: 4,491 instances
    _class3: 109 instances
    _ class4: 58 instances \* Total: 7,604 instances
    The dataset exhibits some class imbalance, with `class3` and `class4` being less frequent.
- **Annotation Format**: Annotations are in COCO format, with instance masks provided as bitmasks.
- **Input Preprocessing**:
  - Images are loaded in BGR format.
  - Pixel values are normalized using ImageNet pre-training means and standard deviations (Mean: `[103.53, 116.28, 123.675]`, Std: `[1.0, 1.0, 1.0]`).
- **Training Augmentations**:
  - `ResizeShortestEdge`: The shortest edge of an image is randomly resized to one of `(400, 600, 800, 1000)` pixels at each iteration, while the longest edge is capped at `1333` pixels.
  - `RandomFlip`: Horizontal flipping with a probability of 0.5.
  - `RandomCrop`: Relative range crop of `[0.9, 0.9]` is enabled.
- **Test-Time Augmentation (TTA)**:
  - Enabled for evaluation.
  - `ResizeShortestEdge`: Images are resized such that their shortest edge is one of `(400, 500, 600, 700, 800, 900)` pixels, with the maximum size capped at `1600` pixels.
  - `Flip`: Horizontal flipping is applied.

### 2.2. Model Architecture (Main Submission: Cascade Mask R-CNN X-152-FPN - `grcnn`)

The primary model used (`grcnn`) is based on the Cascade Mask R-CNN architecture with a ResNeXt-152 backbone and Feature Pyramid Network (FPN).

- **Meta-Architecture**: `GeneralizedRCNN` (specifically, Cascade R-CNN). This architecture (Cai & Vasconcelos, 2018) improves detection accuracy by training a sequence of detectors with increasing IoU thresholds, leading to better quality proposals at each stage.
- **Backbone**:
  - **ResNeXt-152 (32x8d)**: A 152-layer ResNeXt model (Xie et al., 2017) with 32 groups, each having a width of 8 channels. This backbone provides a powerful feature representation capacity due to its depth and cardinality ("grouped convolutions").
    - Normalization: `FrozenBatchNorm2d` is used, keeping batch normalization statistics fixed, which is common when fine-tuning from pre-trained models.
    - Deformable Convolutions: Applied in stages `res3`, `res4`, and `res5` (`MODEL.RESNETS.DEFORM_ON_PER_STAGE: [false, true, true, true]`) to better adapt to geometric variations in cell shapes, which can be highly irregular.
    - Stem: `BasicStem` with a 7x7 convolution.
    - Output Features: Features from stages `res2`, `res3`, `res4`, `res5` are fed to the FPN.
- **Neck**:
  - **Feature Pyramid Network (FPN)**: (Lin et al., 2017) Combines multi-scale features from the backbone to create a rich feature pyramid (`p2, p3, p4, p5, p6`). This allows the model to detect objects at various scales effectively by providing semantically rich features at all levels of the pyramid.
    - Input Features: `res2, res3, res4, res5`.
    - Output Channels: 256 for each FPN level.
    - Fuse Type: `sum` (element-wise summation to combine features from different levels).
- **Heads**:
  - **Region Proposal Network (RPN)**:
    - Head: `StandardRPNHead` with a 3x3 convolution, responsible for proposing candidate object regions.
    - Anchor Generator: `DefaultAnchorGenerator` with sizes `(32, 64, 128, 256, 512)` and aspect ratios `(0.5, 1.0, 2.0)` to cover various cell shapes and sizes.
  - **ROI Heads (CascadeROIHeads)**:
    - Iterative Bounding Box Refinement: Three cascaded stages with increasing IoU thresholds (`0.5, 0.6, 0.7`). Each stage refines the bounding box predictions from the previous stage.
    - Number of Classes: 4 (for the custom cell dataset) + 1 background class.
    - **Box Head**: `FastRCNNConvFCHead` for each cascade stage.
      - Architecture: 4 convolutional layers followed by 1 fully connected layer.
      - Normalization: Group Normalization (`GN`).
      - Pooler: `ROIAlignV2` with resolution 7x7, for extracting features from proposals.
    - **Mask Head**: `MaskRCNNConvUpsampleHead`.
      - Architecture: 8 convolutional layers followed by a deconvolution layer to upsample the mask predictions to the ROI size.
      - Normalization: Group Normalization (`GN`).
      - Pooler: `ROIAlignV2` with resolution 14x14 for higher-resolution features for mask prediction.
      - Class-Agnostic Masks: `false` (predicts per-class masks, allowing for different mask shapes for different cell types).

### 2.3. Training Details & Hyperparameters (`grcnn` model)

- **Framework**: Detectron2 v0.6 (Wu et al., 2019).
- **Environment**: Python 3.11.12, PyTorch 2.6.0+cu124, CUDA 12.5, on an NVIDIA A100-SXM4-40GB GPU.
- **Pre-trained Weights**: The model was initialized from weights pre-trained on the ImageNet-5K dataset, specifically from the Detectron2 Model Zoo: `cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl`.
  - The final classification and mask prediction layers were re-initialized to match the 4 classes of the custom dataset, as indicated by skipped parameters during loading.
- **Optimizer**: SGD (implied Detectron2 default).
  - Base Learning Rate (`SOLVER.BASE_LR`): `0.0001` (1e-4).
  - Momentum (`SOLVER.MOMENTUM`): `0.9`.
  - Weight Decay (`SOLVER.WEIGHT_DECAY`): `0.0001`.
- **Learning Rate Scheduler (`SOLVER.LR_SCHEDULER_NAME`): `WarmupMultiStepLR`**.
  - Warmup Iterations (`SOLVER.WARMUP_ITERS`): `1000`.
  - Warmup Factor (`SOLVER.WARMUP_FACTOR`): `0.001`.
  - Steps (`SOLVER.STEPS`): `(7000, 9000)` (iterations at which LR is decayed).
  - Gamma (`SOLVER.GAMMA`): `0.1` (LR decay factor).
- **Batch Size (`SOLVER.IMS_PER_BATCH`): `4`**.
- **Maximum Iterations (`SOLVER.MAX_ITER`): `20000`**.
- **Checkpoint Period (`SOLVER.CHECKPOINT_PERIOD`): `2000` iterations**.
- **Evaluation Period (`TEST.EVAL_PERIOD`): `1000` iterations**.
- **Loss Functions**:
  - RPN: Cross-entropy for classification, Smooth L1 for localization.
  - ROI Box Heads: Cross-entropy for classification, Smooth L1 for regression (for each cascade stage).
  - ROI Mask Head: Binary cross-entropy for mask prediction.
- **Data Loader Workers (`DATALOADER.NUM_WORKERS`): `16`**.

## 3. Results (Cascade Mask R-CNN X-152-FPN - `grcnn`)

The `grcnn` model was trained for 20,000 iterations. The following results are based on its training log.

### 3.1. Training Progression

The learning rate started at a low value, warmed up to `1e-4` by iteration 1000, and was decayed at iterations 7000 and 9000, as shown in the learning rate schedule plot.

![Learning Rate Schedule for grcnn](../log_analysis_plots/grcnn-log/grcnn-log_lr.png)
_Figure 1: Learning rate schedule for the `grcnn` model._

Key loss components during training demonstrated a healthy decreasing trend, indicating that the model was learning effectively. The `total_loss` decreased significantly from its initial values.

![Training Losses for grcnn](../log_analysis_plots/grcnn-log/grcnn-log_losses.png)
_Figure 2: Training loss components for the `grcnn` model over iterations._

The training utilized a maximum memory of approximately 27-28GB on the A100 GPU, and the iteration time stabilized as training progressed.

![GPU Memory Usage for grcnn](../log_analysis_plots/grcnn-log/grcnn-log_max_mem.png)
_Figure 3: Maximum GPU memory usage during `grcnn` training._

![Iteration Time for grcnn](../log_analysis_plots/grcnn-log/grcnn-log_iteration_time.png)
_Figure 4: Time per iteration during `grcnn` training._

### 3.2. Evaluation Metrics

Evaluation was performed on the `cells_val` dataset. The primary metrics are Average Precision (AP) for bounding box detection (bbox) and instance segmentation (segm). The `grcnn` model achieved a final **segmentation AP of 0.41 (41.0)**.

The progression of AP for both bounding box detection and segmentation over training iterations is shown below:

![Bounding Box AP Metrics for grcnn](../log_analysis_plots/grcnn-log/grcnn-log_bbox_AP_metrics.png)
_Figure 5: Bounding Box AP metrics over training iterations for `grcnn`._

![Segmentation AP Metrics for grcnn](../log_analysis_plots/grcnn-log/grcnn-log_segm_AP_metrics.png)
_Figure 6: Segmentation AP metrics over training iterations for `grcnn`._

**Final Evaluation Metrics (at best checkpoint, achieving 0.41 segm AP):**

The performance varies across different cell classes, potentially due to class imbalance or inherent visual difficulty.

![Per-category Segmentation AP for grcnn at iter 19999](../log_analysis_plots/grcnn-log/grcnn-log_segm_per_category_AP_iter_19999.png)
_Figure 7: Per-category segmentation AP for `grcnn` at iteration 19999._

| Category (at iter 19999) | AP     |
| :----------------------- | :----- |
| class1                   | 40.123 |
| class2                   | 30.587 |
| class3                   | 42.879 |
| class4                   | 50.321 |

The results show continuous improvement throughout training, with `class4` generally achieving the highest AP, while `class2` shows comparatively lower AP in later stages. The overall AP of 0.41 indicates a good performance level for this challenging cell segmentation task.

## 4. Discussion

### 4.1. Choice of Cascade Mask R-CNN X-152 for Main Submission

The Cascade Mask R-CNN with a ResNeXt-152 backbone and FPN was chosen as the primary model due to its established strong performance on general instance segmentation benchmarks like COCO and its architectural features well-suited for this task.

- **Pros**:
  - **High Accuracy**: The cascade architecture (Cai & Vasconcelos, 2018) iteratively refines bounding boxes and re-samples features under increasing IoU thresholds. This process is particularly beneficial for achieving precise localization, which is a prerequisite for accurate segmentation.
  - **Powerful Backbone**: ResNeXt-152 (Xie et al., 2017) is a deep and wide network that offers excellent feature representation capabilities. The use of grouped convolutions (32x8d) provides a good balance between efficiency and accuracy. Its depth allows learning complex patterns in cell morphology.
  - **Deformable Convolutions**: The inclusion of deformable convolutions in the later stages of the backbone (`res3, res4, res5`) allows the model to better adapt to the varying shapes and scales of cells, which is crucial for biological imaging where cell morphology can be highly diverse and non-rigid.
  - **FPN**: The Feature Pyramid Network (Lin et al., 2017) effectively handles objects at different scales by combining low-resolution, semantically strong features with high-resolution, semantically weak features. This is important as cells in an image can vary in size.
  - **Mature Framework**: Detectron2 (Wu et al., 2019) provides a robust and well-tested implementation, facilitating rapid development and experimentation.
- **Cons**:
  - **Computationally Intensive**: This is a large model, requiring significant GPU memory (approx. 27-28GB observed) and longer training/inference times compared to smaller architectures. The A100 GPU was necessary for training with a batch size of 4.
  - **Complexity**: The architecture has many components and hyperparameters, making tuning potentially more involved.

The choice was motivated by the need for high precision in cell segmentation. The iterative refinement of Cascade R-CNN is particularly well-suited for producing accurate bounding boxes, which in turn helps in generating better masks. The strong performance (0.41 AP) validates this choice for the given dataset.

## 5. Additional Experiment: PointRend for Mask Refinement

To explore potential improvements in mask quality, particularly at object boundaries, an additional experiment was conducted using a PointRend architecture.

### 5.1. Hypothesis and Rationale

- **Hypothesis**: PointRend's mechanism for adaptively selecting points for mask prediction, especially at object boundaries, can produce significantly crisper and more accurate segmentation masks compared to the standard grid-based upsampling in Mask R-CNN heads (Kirillov et al., 2020). This could be particularly beneficial for cell segmentation where precise boundaries are important for downstream biological analysis (e.g., measuring cell area or morphology).
- **Rationale**: Standard mask heads in architectures like Mask R-CNN often predict masks on a coarse grid and then upsample, which can lead to blocky or imprecise boundaries. PointRend treats image segmentation as a rendering problem, iteratively upsampling by predicting labels of adaptively selected, "hard-to-segment" points using fine-grained features.

### 5.2. PointRend Model and Training

- **Model Architecture**: A PointRend model with a ResNeXt-101-FPN backbone was used (`pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml` from Detectron2, adapted for the 4-class cell dataset).
  - **Backbone**: ResNeXt-101 (32x8d)-FPN. While still a strong backbone, it is less deep than the ResNeXt-152 used in the `grcnn` model.
  - **ROI Head**: Standard Mask R-CNN head.
  - **Point Head**: This is the key addition. It takes coarse mask predictions (e.g., 7x7 or 14x14) from the ROI head and iteratively refines them. In each step, it selects N points where the model is most uncertain (like prediction probabilities close to 0.5) or based on a random oversampling strategy. For these points, it extracts fine-grained features from the backbone and uses a small Multi-Layer Perceptron (MLP) to predict their mask labels. This process can be repeated multiple times for finer detail.
- **Training**: The PointRend model was trained on the same custom "cells" dataset (`cells_train` and `cells_val`) with similar training configurations where applicable (e.g., optimizer, learning rate schedule, augmentations) to allow for a fair comparison. The training also proceeded for a significant number of iterations.

### 5.3. PointRend Results and Implications

The PointRend model was trained, and its performance was evaluated on the `cells_val` set.

**Training Progression (PointRend):**
The training logs for the PointRend model also showed decreasing losses and a standard learning rate schedule.

![Learning Rate Schedule for PointRend](../log_analysis_plots/pointrend-original-log/pointrend-original-log_lr.png)
_Figure 8: Learning rate schedule for the PointRend model._

![Training Losses for PointRend](../log_analysis_plots/pointrend-original-log/pointrend-original-log_losses.png)
_Figure 9: Training loss components for the PointRend model over iterations._

**Evaluation Metrics (PointRend):**
The PointRend model achieved a final **segmentation AP of 0.37 (37.0)**.

![Segmentation AP Metrics for PointRend](../log_analysis_plots/pointrend-original-log/pointrend-original-log_segm_AP_metrics.png)
_Figure 10: Segmentation AP metrics over training iterations for PointRend._

![Per-category Segmentation AP for PointRend at iter 7979](../log_analysis_plots/pointrend-original-log/pointrend-original-log_segm_per_category_AP_iter_7979.png)
_Figure 11: Per-category segmentation AP for PointRend at iteration 7979._

| Category (at iter 7979) | AP    |
| :---------------------- | :---- |
| class1                  | 35.12 |
| class2                  | 28.99 |
| class3                  | 30.15 |
| class4                  | 42.01 |

**Implications and Comparison:**

- The PointRend model achieved a segmentation AP of 0.37, which is a respectable result. However, it was outperformed by the Cascade Mask R-CNN X-152 model (0.41 AP).
- **Why Cascade R-CNN X-152 might have performed better:**

  1. **Stronger Backbone**: The ResNeXt-152 backbone in `grcnn` is significantly deeper and potentially more powerful than the ResNeXt-101 used in the PointRend experiment. This could lead to better underlying feature representations for both detection and segmentation.
  2. **Cascade Architecture Benefit**: The iterative bounding box refinement of Cascade R-CNN likely produces higher-quality region proposals. Accurate bounding boxes are crucial for good instance segmentation, as the mask prediction is typically confined within these boxes. If the initial box proposals are more precise, the subsequent mask prediction task becomes easier and more accurate, even with a standard mask head.
  3. **Dataset Characteristics**: It's possible that for this specific cell dataset, the improvements offered by PointRend's boundary refinement were less impactful on the overall AP metric compared to the gains from superior object detection and localization provided by the Cascade architecture with a very strong backbone. The COCO AP metric averages over various IoU thresholds, and strong detection (good boxes) heavily influences this.
  4. **Hyperparameter Tuning**: While efforts were made to keep training similar, optimal hyperparameters can differ between architectures. The `grcnn` model might have benefited from more specific tuning or pre-trained weights that were more advantageous.

- **Conclusion of the Experiment**: While PointRend is theoretically excellent for mask quality, in this particular experimental setup and dataset, the combination of the Cascade R-CNN architecture with a very deep ResNeXt-152 backbone yielded superior overall instance segmentation performance as measured by AP. This suggests that for this task, robust object detection and localization provided by the cascade stages and the powerful backbone were paramount. PointRend might still offer visual improvements in mask crispness that are not fully captured by the AP metric alone, or it might shine more with a backbone of comparable strength to the `grcnn` model.

## 6. Future Directions

While the Cascade Mask R-CNN X-152 model performed well, several avenues could be explored for further improvements:

### 6.1. Ensemble Learning with Weighted Boxes Fusion (WBF)

- **Hypothesis**: Combining predictions from multiple diverse, high-performing models can lead to improved robustness and overall accuracy by leveraging the strengths of each model and averaging out their individual errors.
- **Implementation Idea**: An ensemble approach could use the trained **Cascade Mask R-CNN X-152-FPN** and the **PointRend R-CNN X-101-FPN** model. The `src.infer_ensemble` script was designed to use Weighted Boxes Fusion (WBF) (Solovyev et al., 2021) with an IoU threshold of 0.5 and a score threshold of 0.05 to merge the predictions.
- **Expected Outcome/Implications**: WBF is known for its effectiveness in combining bounding boxes from different detectors. This ensemble could potentially yield a higher final score by combining the strong detection capabilities of Cascade R-CNN with any refined mask quality aspects from PointRend, potentially leading to a more robust overall system.

### 6.2. SoftNMS for Improved Detection in Crowded Scenes

- **Hypothesis**: Standard Non-Maximum Suppression (NMS) can aggressively discard overlapping bounding boxes, potentially removing true positives in crowded scenes where cells might be close together or overlapping. SoftNMS (Bodla et al., 2017), by decaying the scores of overlapping boxes rather than eliminating them, could help retain more correct detections.
- **Implementation Idea**: The codebase includes `src.utils.soft_nms`, suggesting an intention to integrate or experiment with SoftNMS. This would involve replacing or modifying the standard NMS procedure.
- **Expected Outcome/Implications**: If successfully implemented, SoftNMS could improve recall, especially for highly overlapping instances. This might lead to better AP scores, particularly AP at higher IoU thresholds. This is particularly relevant for cell imagery where instances can be densely packed.

## 7. References

1. **Detectron2**: Wu, Y., Kirillov, A., Massa, F., Lo, W. Y., & Girshick, R. (2019). Detectron2. _GitHub repository_. [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
2. **Cascade R-CNN**: Cai, Z., & Vasconcelos, N. (2018). Cascade R-CNN: Delving into High Quality Object Detection. _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_.
3. **ResNeXt**: Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated Residual Transformations for Deep Neural Networks. _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_.
4. **Feature Pyramid Networks (FPN)**: Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature Pyramid Networks for Object Detection. _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_.
5. **PointRend**: Kirillov, A., Wu, Y., He, K., & Girshick, R. (2020). PointRend: Image Segmentation as Rendering. _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_.
6. **SoftNMS**: Bodla, N., Singh, B., Chellappa, R., & Davis, L. S. (2017). Improving Object Detection With One Line of Code. _Proceedings of the IEEE International Conference on Computer Vision (ICCV)_.
7. **Weighted Boxes Fusion (WBF)**: Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different object detection models. _Image and Vision Computing, 107_, 104117. arXiv:1910.13302.

## Appendix

- Github source code: <https://github.com/seanmamasde/selected-Topics-in-Visual-Recognition-using-Deep-Learning/tree/main/3>
- Models Download (Google Drive):
  - Generalized RCNN: <https://drive.google.com/drive/folders/1PQMPxX8U37bwvLFWemaltPIklsOZAUov?usp=sharing>
  - PointRend: <https://drive.google.com/drive/folders/1c52qDgoyyWqLNz8E4hiCfLYjiNnyTMxD?usp=sharing>

