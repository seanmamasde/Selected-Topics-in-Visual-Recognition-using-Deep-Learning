import argparse
import json
from pathlib import Path

import cv2
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
from detectron2.modeling import build_model
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes, Instances
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

import src.utils.soft_nms
from src.config import CASCADE_YAML, DATA_ROOT, NUM_CLASSES
from src.dataset import register_cell_dataset
from src.utils.rle import encode_binary_mask


def load_model(cfg_path, weights_path, device, type="grcnn"):
    cfg = get_cfg()
    if type == "pointrend":
        add_pointrend_config(cfg)
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = NUM_CLASSES
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 1024  # Match training/other inference
    cfg.MODEL.DEVICE = device
    # Set thresholds for individual model inference *before* ensembling NMS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.01  # Very low threshold initially
    )
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7  # High threshold initially
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights_path)
    model.eval()
    return model


def combine_predictions(
    predictions_list, iou_threshold=0.5, score_threshold=0.05, method="nms"
):
    """Combine predictions from multiple models via NMS, Soft‑NMS, or WBF."""
    all_boxes, all_scores, all_classes, all_masks = [], [], [], []
    for pred in predictions_list:
        inst = pred["instances"]  # your dict from model(inputs)[0]
        all_boxes.append(inst.pred_boxes.tensor)
        all_scores.append(inst.scores)
        all_classes.append(inst.pred_classes)
        all_masks.append(inst.pred_masks)

    # Flatten for final indexing
    cat_boxes = torch.cat(all_boxes, dim=0)
    cat_scores = torch.cat(all_scores, dim=0)
    cat_classes = torch.cat(all_classes, dim=0)
    cat_masks = torch.cat(all_masks, dim=0)

    # Perform NMS, Soft‑NMS, or Weighted Boxes Fusion
    if method == "nms":
        from detectron2.layers import batched_nms

        keep_indices, _ = batched_nms(
            cat_boxes, cat_scores, cat_classes, iou_threshold
        )
        final_scores = cat_scores[keep_indices]
        score_keep = final_scores > score_threshold
        final_indices = keep_indices[score_keep]

    elif method == "soft_nms":
        keep_indices, _ = src.utils.soft_nms.batched_soft_nms(
            cat_boxes,
            cat_scores,
            cat_classes,
            linear_threshold=iou_threshold,
            prune_threshold=score_threshold,
        )
        final_indices = keep_indices

    elif method == "wbf":
        from ensemble_boxes import weighted_boxes_fusion

        # 1) Normalize boxes to [0,1] by image size
        h, w = predictions_list[0]["instances"].image_size
        norm_boxes = []
        for b in all_boxes:
            # b: Tensor[N,4] in x1,y1,x2,y2
            arr = b.clone().cpu().numpy()
            arr[:, [0, 2]] /= w  # x coords
            arr[:, [1, 3]] /= h  # y coords
            norm_boxes.append(arr.tolist())

        scores_list = [s.cpu().numpy().tolist() for s in all_scores]
        labels_list = [c.cpu().numpy().tolist() for c in all_classes]

        # 2) Run WBF
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            norm_boxes,
            scores_list,
            labels_list,
            weights=[1.0] * len(norm_boxes),
            iou_thr=iou_threshold,
            skip_box_thr=score_threshold,
        )

        # 3) Denormalize fused boxes back to pixel coords
        fused_boxes = [
            [
                box[0] * w,  # x1
                box[1] * h,  # y1
                box[2] * w,  # x2
                box[3] * h,  # y2
            ]
            for box in fused_boxes
        ]

        # 4) Map fused boxes back to nearest original index to keep masks
        fused_boxes_tensor = torch.tensor(fused_boxes, device=cat_boxes.device)
        final_indices = []
        for fb in fused_boxes_tensor:
            dists = ((cat_boxes - fb) ** 2).sum(dim=1)
            final_indices.append(int(dists.argmin().item()))
        final_indices = torch.tensor(
            final_indices, dtype=torch.long, device=cat_boxes.device
        )

    else:
        raise ValueError(f"Invalid method: {method!r}")

    # Build final Instances
    final = Instances(predictions_list[0]["instances"].image_size)
    final.pred_boxes = Boxes(cat_boxes[final_indices])
    final.scores = cat_scores[final_indices]
    final.pred_classes = cat_classes[final_indices]
    final.pred_masks = cat_masks[final_indices]
    return final


def _id_map():
    # (Same as in infer.py)
    with open(DATA_ROOT / "test_image_name_to_ids.json") as f:
        return {d["file_name"]: d["id"] for d in json.load(f)}


def _test_imgs():
    # (Same as in infer.py)
    return sorted((DATA_ROOT / "test_release").glob("*.tif"))


def main(args):
    register_cell_dataset()
    device = "cuda"

    # for cfg_path, weight_path in zip(args.cfgs, args.weights):
    #     print(f"Loading model with config={cfg_path} and weights={weight_path}")
    #     model, _ = load_model(cfg_path, weight_path, device)  # unchanged loader
    #     models.append(model)
    models = [
        load_model(
            "/content/detectron2/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml",
            "output1/model_0003999.pth",
            device,
        ),
        load_model(
            "/content/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml",
            "output/model_0003999.pth",
            device,
            "pointrend",
        ),
    ]

    id_map = _id_map()
    results = []

    for pth in tqdm(_test_imgs()):
        img_bgr = cv2.imread(str(pth))
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.as_tensor(
            img_rgb.astype("float32").transpose(2, 0, 1)
        ).to(device)
        inputs = [{"image": img_tensor, "height": h, "width": w}]

        # Get predictions from each model
        predictions_list = []
        with torch.no_grad():
            for model in models:
                preds = model(inputs)[0]  # Get output dict for the first image
                predictions_list.append(preds)

        # Combine predictions using NMS or Soft-NMS
        final_inst = combine_predictions(
            predictions_list,
            iou_threshold=args.iou_thresh,
            score_threshold=args.score_thresh,
            method=args.nms_method,
        ).to("cpu")

        # Format results (similar to infer.py)
        for box, mask, cls, score in zip(
            final_inst.pred_boxes.tensor.numpy(),
            final_inst.pred_masks.numpy(),
            final_inst.pred_classes.numpy(),
            final_inst.scores.numpy(),
        ):
            x1, y1, x2, y2 = box.tolist()
            rle = encode_binary_mask(mask)
            results.append(
                {
                    "image_id": int(id_map[pth.name]),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(score),
                    "category_id": int(cls) + 1,
                    "segmentation": {
                        "size": [int(h), int(w)],
                        "counts": rle["counts"],
                    },
                }
            )

    # Save results (similar to infer.py)
    out_zip = Path(args.output)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_zip.parent / "test-results-ensemble.json"
    with open(json_path, "w") as f:
        json.dump(results, f)
    import zipfile

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="test-results.json")
    print(f"Ensemble submission written → {out_zip}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="submission_ensemble.zip")
    ap.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for final NMS/Soft-NMS",
    )
    ap.add_argument(
        "--score-thresh",
        type=float,
        default=0.05,
        help="Score threshold for final predictions",
    )
    ap.add_argument(
        "--nms-method",
        default="wbf",
        choices=["nms", "soft_nms", "wbf"],
        help="Method for combining predictions",
    )
    main(ap.parse_args())
