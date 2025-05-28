#!/usr/bin/env python3
"""
Plot two training figures from VRDL-HW4 training.log
---------------------------------------------------
  • Figure 1 (train_curves.png)
        - rolling-mean train loss  (blue line)
        - ±1 sigma band as translucent shadow
        - learning-rate scatter    (orange)
  • Figure 2 (val_curves.png)
        - val L1 (blue)  & val PSNR (orange) dual-axis
  • Two black dashed lines mark the three training phases
Usage
-----
    python plot_training_curves.py training.log
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Regex helpers
RE_TRAIN = re.compile(
    r"Epoch\s+(\d+)/\d+,\s+Train Loss:\s+([-0-9.]+),\s+Learning Rate:\s+([0-9.e-]+)")
RE_VAL = re.compile(
    r"Epoch\s+(\d+)/\d+,\s+Val L1:\s+([0-9.]+),\s+Val PSNR:\s+([0-9.]+)")
RE_SPLIT = re.compile(r"^---+$")    # phase separator


def parse_log(path: Path):
    tr_e, tr_loss, tr_lr = [], [], []
    va_e, va_l1, va_psnr = [], [], []
    splits = []

    with path.open() as fh:
        for line in fh:
            if m := RE_TRAIN.search(line):
                tr_e.append(int(m.group(1)))
                tr_loss.append(float(m.group(2)))
                tr_lr.append(float(m.group(3)))
            elif m := RE_VAL.search(line):
                va_e.append(int(m.group(1)))
                va_l1.append(float(m.group(2)))
                va_psnr.append(float(m.group(3)))
            elif RE_SPLIT.match(line.strip()) and tr_e:
                # next epoch starts new phase
                splits.append(tr_e[-1] + 1)

    return (pd.DataFrame(dict(epoch=tr_e, loss=tr_loss, lr=tr_lr)),
            pd.DataFrame(dict(epoch=va_e, l1=va_l1, psnr=va_psnr)),
            splits[:2])                              # keep exactly two


def xticks(max_ep, step=50):
    return np.arange(0, ((max_ep + step - 1)//step)*step + step, step)


sns.set_theme(style="darkgrid")                      # dark background
BLUE, ORANGE = "#1f77b4", "#ff7f0e"
LIGHT_BLUE, LIGHT_ORANGE = "#aec7e8", "#ffbb78"


# helper: draw two differently styled grids
def _dual_grid(ax_primary, ax_secondary, color_primary, color_secondary):
    # primary axis: thin solid grid
    ax_primary.grid(True, which="both", color=color_primary, alpha=.5, lw=1.2)
    # secondary axis: turn off seaborn's default, then add dashed gridlines
    ax_secondary.grid(False)
    for loc, spine in ax_secondary.spines.items():
        spine.set_visible(False)  # keep frame minimalist
    # draw custom dashed gridlines that line up with secondary y-ticks
    for y in ax_secondary.get_yticks():
        ax_secondary.axhline(
            y, color=color_secondary, alpha=.5, lw=1.2, ls="--", zorder=0)


def plot_train(df, splits, fname):
    win = 15
    roll = df.loss.rolling(win, center=True)
    mean, std = roll.mean(), roll.std()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # ±1 sigma shadow + smoothed line
    ax1.fill_between(df.epoch, mean-std, mean+std,
                     color=BLUE, alpha=.15, lw=0)
    ax1.plot(df.epoch, mean, color=BLUE, lw=1.4, label="Train loss")

    # LR scatter
    ax2.scatter(df.epoch, df.lr, color=ORANGE, s=1.4, label="LR")
    # ax2.plot(df.epoch, df.lr, color=ORANGE, lw=1.4, label="Learning rate")

    # dual grids ---------------------------------------------------------------
    _dual_grid(ax1, ax2, LIGHT_BLUE, LIGHT_ORANGE)

    # cosmetics
    ax1.set_ylabel("Train loss", color=BLUE)
    ax2.set_ylabel("Learning Rate", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax1.set_xticks(xticks(df.epoch.iat[-1], step=50))
    ax1.set_xlabel("Epoch")

    for x in splits:
        ax1.axvline(x, color="black", ls="--", lw=1)

    ax1.set_title("Training Loss & Learning Rate")
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    print("saved →", fname)


def plot_val(df, splits, fname):
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    ax1.plot(df.epoch, df.l1, color=BLUE, lw=1.4, label="Val L1")
    ax2.plot(df.epoch, df.psnr, color=ORANGE, lw=1.4, label="Val PSNR")

    # dual grids ---------------------------------------------------------------
    _dual_grid(ax1, ax2, LIGHT_BLUE, LIGHT_ORANGE)

    ax1.set_ylabel("Validation L1", color=BLUE)
    ax2.set_ylabel("PSNR (dB)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax1.set_xticks(xticks(df.epoch.iat[-1], step=50))
    ax1.set_xlabel("Epoch")

    for x in splits:
        ax1.axvline(x, color="black", ls="--", lw=1)

    ax1.set_title("Validation Curves")
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    print("saved →", fname)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage  : python plot_training_curves.py training.log")
    log = Path(sys.argv[1])
    tr_df, va_df, phase_pts = parse_log(log)

    plot_train(tr_df, phase_pts, "train_curves.png")
    plot_val(va_df, phase_pts, "val_curves.png")
    plt.show()
