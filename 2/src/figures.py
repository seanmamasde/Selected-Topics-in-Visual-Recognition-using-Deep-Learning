#!/usr/bin/env python3
"""
figures.py  —  create training‑curve PNGs from one or more trainer logs
Usage:
    python figures.py <log1> [log2 ...]
The script produces up to three images in ./figures:
    curves_val_loss.png   (if at least one log has val_loss)
    curves_map.png
    curves_accuracy.png
"""
import re, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def parse_log(path: Path) -> pd.DataFrame:
    txt = path.read_text()

    fp32_rows = [
        (int(ep), float(vloss), float(acc))
        for ep, vloss, acc in re.findall(
            r"\[Epoch\s+(\d+).*?val_loss=([\d.]+).*?acc=([\d.]+)", txt, re.S
        )
    ]

    amp_rows = [
        (int(ep), None, float(acc))
        for ep, acc in re.findall(
            r"\[Epoch\s+(\d+)\]\s+accuracy\s*=\s*([\d.]+)", txt
        )
    ]

    rows = fp32_rows or amp_rows
    df = pd.DataFrame(rows, columns=["epoch", "val_loss", "accuracy"])

    maps = [
        float(x)
        for x in re.findall(
            r"Average Precision\s+\(AP\).*?all.*?=\s+([\d.]+)", txt, re.S
        )
    ]
    # truncate in case there are more AP lines than epochs
    df["map"] = maps[: len(df)]
    return df.set_index("epoch")


def plot_metric(
    metric: str, dfs: dict[str, pd.DataFrame], out_dir: Path
) -> None:
    """Plot one metric for all logs on a shared axis."""
    cmap = get_cmap("Blues")  # light → dark blues
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for idx, (name, df) in enumerate(dfs.items(), start=1):
        if metric not in df or df[metric].dropna().empty:
            continue
        color = cmap(
            0.3 + 0.6 * idx / (len(dfs) + 1)
        )  # spread nicely in the map
        df[metric].plot(
            ax=ax,
            marker="o",
            linewidth=1.3,
            label=name,
            color=color,
        )

    if not ax.lines:  # nothing was plotted
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(metric.replace("_", " ").title())
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"curves_{metric}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python figures.py <log1> [log2 ...]")
        sys.exit(1)

    logs = {Path(p).stem: parse_log(Path(p)) for p in sys.argv[1:]}

    for metric in ("val_loss", "map", "accuracy"):
        plot_metric(metric, logs, Path("figures"))

    print("✓ figures saved to ./figures/")


if __name__ == "__main__":
    main()


"""
import re, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_log(path: Path) -> pd.DataFrame:
    txt = path.read_text()

    # Format A: “[Epoch 10] val_loss=0.2165 | acc=0.5413”
    fp32_rows = [
        (int(ep), float(vloss), float(acc))
        for ep, vloss, acc in re.findall(
            r"\[Epoch\s+(\d+).*?val_loss=([\d.]+).*?acc=([\d.]+)", txt, re.S
        )
    ]

    # Format B: “[Epoch 07] accuracy = 0.8957”
    amp_rows = [
        (int(ep), None, float(acc))
        for ep, acc in re.findall(
            r"\[Epoch\s+(\d+)\]\s+accuracy\s*=\s*([\d.]+)", txt
        )
    ]

    rows = fp32_rows or amp_rows
    df = pd.DataFrame(rows, columns=["epoch", "val_loss", "accuracy"])

    # mAP values from COCO summary block (same in both logs)
    maps = [
        float(x)
        for x in re.findall(
            r"Average Precision\s+\(AP\).*?all.*?=\s+([\d.]+)", txt, re.S
        )
    ]
    df["map"] = maps[: len(df)]  # align list length to epoch count
    return df.set_index("epoch")


def plot_curves(df: pd.DataFrame, title: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["val_loss", "map", "accuracy"]:
        if metric not in df:
            continue
        if df[metric].dropna().empty:  # skip “all‑NaN” columns
            continue
        ax = df[metric].plot(marker="o", linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{title}: {metric.replace('_', ' ').title()}")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out_dir / f"{title.lower()}_{metric}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python figures.py <log1> [log2 ...]")
        sys.exit(1)

    for log in map(Path, sys.argv[1:]):
        df = parse_log(log)
        plot_curves(df, log.stem, Path("figures"))
        print(f"✓ processed {log}")

"""
