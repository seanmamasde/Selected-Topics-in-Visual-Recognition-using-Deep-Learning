import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """IoU for *xyxy* boxes."""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_predictions(
    preds: List[Dict], gts: List[Dict], iou_thr: float = 0.5
) -> float:
    to_xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]
    from collections import defaultdict

    gt_pool: Dict[int, List[Dict]] = defaultdict(list)
    for g in gts:
        gt_pool[g["image_id"]].append(g)

    tp, fp = 0, 0
    for p in preds:
        img_id, cat = p["image_id"], p["category_id"]
        match = None
        for g in gt_pool[img_id]:
            if g["category_id"] != cat:
                continue
            if compute_iou(to_xyxy(p["bbox"]), to_xyxy(g["bbox"])) >= iou_thr:
                match = g
                break
        if match:
            tp += 1
            gt_pool[img_id].remove(match)
        else:
            fp += 1

    return tp / (tp + fp + 1e-8)


def coco_evaluation(
    pred_json_path: Path | str, gt_json_path: Path | str
) -> float:
    coco_gt = COCO(str(gt_json_path))
    coco_dt = coco_gt.loadRes(str(pred_json_path))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    with open(pred_json_path) as f:
        preds = json.load(f)
    gts = [
        dict(
            image_id=a["image_id"],
            bbox=a["bbox"],
            category_id=a["category_id"],
        )
        for a in coco_gt.dataset["annotations"]
    ]
    acc = evaluate_predictions(preds, gts)

    return acc
