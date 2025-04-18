#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F

from seeds import set_seed
from model import get_model
from metrics import coco_evaluation

# ───────────────────── helpers ───────────────────── #


def load_test_images(test_dir: Path) -> list[tuple[int, Path]]:
    """Return [(numeric_id, path), …] *sorted by numeric_id*."""
    ids_paths = []
    for fname in os.listdir(test_dir):
        if fname.endswith(".png"):
            ids_paths.append((int(Path(fname).stem), test_dir / fname))
    return sorted(ids_paths, key=lambda x: x[0])  # numeric order


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    test_dir: Path,
    *,
    save_json_path: Path,
    save_csv_path: Path | None = None,
    score_thresh: float = 0.05,
    device: torch.device | str | None = None,
):
    """Run detector on *test_dir* and dump COCO‑json (+ optional csv)."""
    device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.eval().to(device)

    test_images = load_test_images(test_dir)
    results: list[dict] = []

    print(f"Detecting {len(test_images)} images → {save_json_path.name}")
    for image_id, path in tqdm(test_images, desc="[Infer]"):
        img = Image.open(path).convert("RGB")
        tensor = F.to_tensor(img).to(device)
        outputs = model([tensor])[0]

        for box, label, score in zip(
            outputs["boxes"], outputs["labels"], outputs["scores"]
        ):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append(
                dict(
                    image_id=image_id,
                    category_id=int(label),
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    score=float(score),
                )
            )

    with open(save_json_path, "w") as f:
        json.dump(results, f)
    print("JSON saved to", save_json_path)

    # ---------- CSV (optional) ----------
    if save_csv_path:
        img2pred: dict[int, list[dict]] = defaultdict(list)
        for r in results:
            img2pred[r["image_id"]].append(r)

        rows: list[dict] = []
        for img_id, _ in test_images:  # already numeric‑sorted
            preds = sorted(
                img2pred.get(img_id, []), key=lambda x: x["bbox"][0]
            )
            if preds:
                digits = [str(int(p["category_id"]) - 1) for p in preds]
                pred_label = int("".join(digits))
            else:
                pred_label = -1
            rows.append(dict(image_id=img_id, pred_label=pred_label))

        (
            pd.DataFrame(rows)
            .sort_values("image_id")  # extra safety
            .to_csv(save_csv_path, index=False)
        )
        print("CSV saved to", save_csv_path)


# ───────────────────── CLI / main ───────────────────── #


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["test", "eval"], required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--data-root", default="/content/data/nycu-hw2-data")
    p.add_argument("--out-dir", default="tester_outputs")
    p.add_argument("--score-thr", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    set_seed(42, deterministic=True)

    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=11, device=device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    if args.mode == "test":
        run_inference(
            model,
            Path(args.data_root) / "test",
            save_json_path=out_dir / "pred.json",
            save_csv_path=out_dir / "pred.csv",
            score_thresh=args.score_thr,
            device=device,
        )
    else:  # eval
        json_path = out_dir / "val_pred.json"
        run_inference(
            model,
            Path(args.data_root) / "valid",
            save_json_path=json_path,
            score_thresh=args.score_thr,
            device=device,
        )
        acc = coco_evaluation(json_path, Path(args.data_root) / "valid.json")
        print(f"\nValidation accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
