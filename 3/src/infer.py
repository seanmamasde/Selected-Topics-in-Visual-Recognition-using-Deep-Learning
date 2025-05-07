"""
Generate submission for competition from trained Cascade Mask RCNN.
"""

import argparse
import json
from pathlib import Path

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm

import src.utils.soft_nms
from src.config import CASCADE_YAML, DATA_ROOT, NUM_CLASSES
from src.dataset import register_cell_dataset
from src.utils.rle import encode_binary_mask


def _cfg(weights):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CASCADE_YAML))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    cfg.freeze()
    return cfg


def _id_map():
    with open(DATA_ROOT / "test_image_name_to_ids.json") as f:
        return {d["file_name"]: d["id"] for d in json.load(f)}


def _test_imgs():
    return sorted((DATA_ROOT / "test_release").glob("*.tif"))


def refine_boxes(boxes_list, image_size):
    """Apply a final refinement to boxes based on aspect ratios or size constraints"""
    # boxes_list is assumed to be a list of [x1, y1, x2, y2]
    # image_size is (height, width)
    refined_boxes = []
    for box in boxes_list:
        # Example refinement: Ensure box coordinates are within image bounds
        # Add more sophisticated logic here if needed (e.g., aspect ratio adjustments)
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_size[1], x2)
        y2 = min(image_size[0], y2)

        # Example: Skip boxes that become too small after clipping
        if x2 <= x1 or y2 <= y1:
            # Append original or a placeholder if needed, or handle skipping later
            refined_boxes.append(box)  # Or potentially skip this box entirely
        else:
            refined_boxes.append([x1, y1, x2, y2])

    return refined_boxes


def main(args):
    register_cell_dataset()
    predictor = DefaultPredictor(_cfg(args.weights))
    id_map = _id_map()
    results = []
    for pth in tqdm(_test_imgs()):
        img = cv2.imread(str(pth))
        h, w = img.shape[:2]
        inst = predictor(img)["instances"].to("cpu")
        # Extract boxes before the loop
        pred_boxes_list = inst.pred_boxes.tensor.numpy().tolist()

        # Apply refinement
        refined_boxes_list = refine_boxes(pred_boxes_list, (h, w))

        # Iterate using refined boxes
        # Ensure the number of masks, classes, scores matches the number of refined boxes
        # (If refine_boxes skips boxes, you'll need to adjust indices accordingly)
        # Assuming refine_boxes returns the same number of boxes for simplicity here:
        for i, (mask, cls, score) in enumerate(
            zip(
                inst.pred_masks.numpy(),
                inst.pred_classes.numpy(),
                inst.scores.numpy(),
            )
        ):
            # Use the refined box
            x1, y1, x2, y2 = refined_boxes_list[i]

            # Skip if refinement made the box invalid (example check)
            if x2 <= x1 or y2 <= y1:
                continue

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
    out_zip = Path(args.output)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_zip.parent / "test-results.json"
    with open(json_path, "w") as f:
        json.dump(results, f)
    import zipfile

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="test-results.json")
    print(f"Submission written → {out_zip}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", default="submission.zip")
    main(ap.parse_args())
