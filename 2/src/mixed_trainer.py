#!/usr/bin/env python3
from __future__ import annotations

import argparse, warnings, os
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import CocoDetectionDataset
from model import get_model
from metrics import coco_evaluation
from tester import run_inference
from seeds import set_seed

# ───────── silence noisy deprecation / future notes ────────── #
warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────── helpers ────────────────────────────── #
def train_one_epoch(
    model,
    optim,
    scaler: torch.cuda.amp.GradScaler,
    loader: DataLoader,
    epoch: int,
    device: torch.device,
    max_batches: int | None,
) -> float:
    """Single training epoch (mixed precision)."""
    model.train()
    pbar = tqdm(loader, desc=f"[Train] {epoch:02d}", total=max_batches)
    running, seen = 0.0, 0

    for b_idx, (imgs, targets) in enumerate(loader, 1):
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            loss_dict = model(imgs, targets)
            losses = sum(loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

        running += losses.item()
        seen += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{losses.item():.3f}")

        if max_batches and b_idx >= max_batches:
            break
    pbar.close()
    return running / seen


@torch.no_grad()
def eval_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    """Validation loss (for monitoring only)."""
    model.train()  # keep loss heads active
    pbar = tqdm(loader, desc="[Valid]", total=max_batches)
    running, seen = 0.0, 0

    for b_idx, (imgs, targets) in enumerate(loader, 1):
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            losses = sum(model(imgs, targets).values())

        running += losses.item()
        seen += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{losses.item():.3f}")

        if max_batches and b_idx >= max_batches:
            break
    pbar.close()
    return running / seen


def save_ckpt(model, optim, epoch: int, path: Path) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "epoch": epoch,
        },
        path,
    )


# ─────────────────────── CLI ───────────────────────────────── #
def get_args():
    p = argparse.ArgumentParser("Train Faster-RCNN (mixed-precision, StepLR)")
    p.add_argument("--data-root", default="nycu-hw2-data")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--step-size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=os.cpu_count())
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    return p.parse_args()


# ─────────────────────── main ──────────────────────────────── #
def main() -> None:
    args = get_args()
    set_seed(42, deterministic=False)  # fast kernels, reproducible RNG

    out_dir = Path(
        args.output_dir or f"outputs/{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # datasets -----------------------------------------------------------
    to_tensor = lambda img, tgt: (F.to_tensor(img), tgt)
    make_ds = lambda split: CocoDetectionDataset(
        Path(args.data_root) / split,
        Path(args.data_root) / f"{split}.json",
        transforms=to_tensor,
    )
    collate = lambda b: tuple(zip(*b))

    train_loader = DataLoader(
        make_ds("train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        make_ds("valid"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # model & optimisation ----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=11, device=device)

    optim = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.StepLR(
        optim, step_size=args.step_size, gamma=args.gamma
    )
    scaler = torch.cuda.amp.GradScaler()

    # training loop ------------------------------------------------------
    best_acc = 0.0  # ← track BEST ACCURACY instead of lowest loss
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model,
            optim,
            scaler,
            train_loader,
            epoch,
            device,
            args.max_train_batches,
        )
        eval_epoch(model, val_loader, device, args.max_val_batches)
        sched.step()

        # --- accuracy on validation set (used for "best") ---------------
        val_json = out_dir / f"val_pred_e{epoch}.json"
        run_inference(
            model,
            Path(args.data_root) / "valid",
            save_json_path=val_json,
            device=device,
            score_thresh=0.7,
        )
        acc = coco_evaluation(val_json, Path(args.data_root) / "valid.json")
        print(f"[Epoch {epoch:02d}] accuracy = {acc:.4f}")

        # --- save checkpoint every epoch --------------------------------
        ckpt_path = out_dir / f"model_e{epoch}.pth"
        save_ckpt(model, optim, epoch, ckpt_path)

        # --- update BEST based on accuracy ------------------------------
        if acc > best_acc:
            best_acc = acc
            print("best-accuracy model updated")
            (out_dir / "best_model.pth").unlink(missing_ok=True)
            ckpt_path.rename(out_dir / "best_model.pth")


if __name__ == "__main__":
    main()
