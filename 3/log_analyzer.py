# Filepath: log_analyzer.py
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Regex patterns tailored for the provided pointrend-original-log.txt
# NOTE: If your other log files have different loss names (e.g., for Cascade R-CNN like loss_cls_stage0),
# you will need to adjust this TRAIN_LOG_PATTERN or create multiple patterns.
TRAIN_LOG_PATTERN = re.compile(
    r"iter:\s*(\d+)\s+"  # Group 1: iter
    r"total_loss:\s*([\d.]+)\s+"  # Group 2: total_loss
    r"loss_cls:\s*([\d.]+)\s+"  # Group 3: loss_cls
    r"loss_box_reg:\s*([\d.]+)\s+"  # Group 4: loss_box_reg
    r"loss_mask:\s*([\d.]+)\s+"  # Group 5: loss_mask
    r"(?:loss_mask_point:\s*([\d.]+)\s+)?"  # Optional Group for loss_mask_point, inner is Group 6
    r"loss_rpn_cls:\s*([\d.]+)\s+"  # Group 7: loss_rpn_cls
    r"loss_rpn_loc:\s*([\d.]+)\s+"  # Group 8: loss_rpn_loc
    r".*?time:\s*([\d.]+)\s+"  # Group 9: time
    r".*?data_time:\s*([\d.]+)\s+"  # Group 10: data_time
    r".*?lr:\s*([\d.e+-]+)\s+"  # Group 11: lr
    r"max_mem:\s*(\d+)M"  # Group 12: max_mem
)

EVAL_HEADER_PATTERN = re.compile(
    r"\|\s*AP\s*\|\s*AP50\s*\|\s*AP75\s*\|\s*APs\s*\|\s*APm\s*\|\s*APl\s*\|"
)
EVAL_VALUES_PATTERN = re.compile(
    r"\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|"
)
PER_CATEGORY_LINE_PATTERN = re.compile(
    r"\|\s*(.+?)\s*\|\s*([\d.]+)\s*(?:\|\s*(.+?)\s*\|\s*([\d.]+)\s*)?(?:\|\s*(.+?)\s*\|\s*([\d.]+)\s*)?\|"
)


def parse_log_file(log_path):
    training_data = []
    evaluation_data = []
    per_category_data = []

    current_iter_for_eval = None
    eval_type = None
    parsing_eval_values_triggered = (
        False  # To check if we are in an eval block and expect values
    )
    lines_after_eval_header = 0  # Counter for lines after header
    parsing_per_category = False

    with open(log_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        train_match = TRAIN_LOG_PATTERN.search(line)
        if train_match:
            iter_val = int(train_match.group(1))
            # Use the iteration number from the training log just before eval, or from checkpoint name
            # The log shows eval happens after a checkpoint, e.g. iter 999, then eval for model_0000999.pth
            current_iter_for_eval = iter_val

            loss_data = {
                "iter": iter_val,
                "total_loss": float(train_match.group(2)),
                "loss_cls": float(train_match.group(3)),
                "loss_box_reg": float(train_match.group(4)),
                "loss_mask": float(train_match.group(5)),
                # loss_mask_point will be added below
                "loss_rpn_cls": float(train_match.group(7)),
                "loss_rpn_loc": float(train_match.group(8)),
                "time": float(train_match.group(9)),
                "data_time": float(train_match.group(10)),
                "lr": float(train_match.group(11)),
                "max_mem": int(train_match.group(12)),
            }

            # Handle optional loss_mask_point (group 6)
            loss_mask_point_val = train_match.group(6)
            if loss_mask_point_val is not None:
                loss_data["loss_mask_point"] = float(loss_mask_point_val)
            else:
                loss_data["loss_mask_point"] = (
                    pd.NA
                )  # Use pandas NA for missing values

            training_data.append(loss_data)

            # Reset eval parsing states if we hit a new training log
            parsing_eval_values_triggered = False
            parsing_per_category = False
            eval_type = None
            lines_after_eval_header = 0

        # Check for start of evaluation blocks
        if "Evaluation results for bbox:" in line:
            eval_type = "bbox"
            parsing_eval_values_triggered = False
            parsing_per_category = False
            lines_after_eval_header = 0
        elif "Evaluation results for segm:" in line:
            eval_type = "segm"
            parsing_eval_values_triggered = False
            parsing_per_category = False
            lines_after_eval_header = 0

        # Parse overall AP values
        if eval_type and EVAL_HEADER_PATTERN.search(line):
            parsing_eval_values_triggered = True
            lines_after_eval_header = 0  # Reset counter for this new header
            continue  # Header found, next lines are separator and data

        if parsing_eval_values_triggered and eval_type:
            lines_after_eval_header += 1
            if (
                lines_after_eval_header == 2
            ):  # Data line is 2 lines after header
                values_match = EVAL_VALUES_PATTERN.search(line)
                if values_match and current_iter_for_eval is not None:
                    evaluation_data.append(
                        {
                            "iter": current_iter_for_eval,
                            "type": eval_type,
                            "AP": (
                                float(values_match.group(1))
                                if values_match.group(1) != "nan"
                                else None
                            ),
                            "AP50": (
                                float(values_match.group(2))
                                if values_match.group(2) != "nan"
                                else None
                            ),
                            "AP75": (
                                float(values_match.group(3))
                                if values_match.group(3) != "nan"
                                else None
                            ),
                            "APs": (
                                float(values_match.group(4))
                                if values_match.group(4) != "nan"
                                else None
                            ),
                            "APm": (
                                float(values_match.group(5))
                                if values_match.group(5) != "nan"
                                else None
                            ),
                            "APl": (
                                float(values_match.group(6))
                                if values_match.group(6) != "nan"
                                else None
                            ),
                        }
                    )
                parsing_eval_values_triggered = False  # Done with this block
                lines_after_eval_header = 0

        # Parse per-category AP values
        if eval_type and f"Per-category {eval_type} AP:" in line:
            parsing_per_category = True
            continue

        if (
            parsing_per_category
            and eval_type
            and current_iter_for_eval is not None
        ):
            if (
                "category" in line.lower()
                and "ap" in line.lower()
                and "|" in line
                and ":" not in line
            ):  # Heuristic for header/separator
                continue

            category_match = PER_CATEGORY_LINE_PATTERN.search(line)
            if category_match:
                groups = category_match.groups()
                for j in range(0, len(groups), 2):
                    if (
                        groups[j]
                        and groups[j + 1]
                        and groups[j].strip()
                        and groups[j].strip().lower()
                        not in ["category", "total", ""]
                    ):
                        cat_name = groups[j].strip()
                        cat_ap = float(groups[j + 1])
                        per_category_data.append(
                            {
                                "iter": current_iter_for_eval,
                                "type": eval_type,
                                "category": cat_name,
                                "AP": cat_ap,
                            }
                        )
            elif (
                not line.strip().startswith("|") and parsing_per_category
            ):  # End of per-category table
                parsing_per_category = False
                # eval_type = None # Keep eval_type until a new block or training log starts

    training_df = pd.DataFrame(training_data)
    evaluation_df = pd.DataFrame(evaluation_data)
    per_category_df = pd.DataFrame(per_category_data)

    return training_df, evaluation_df, per_category_df


def plot_training_metrics(df, output_dir, log_filename_prefix):
    if df.empty:
        print(f"No training data to plot for {log_filename_prefix}.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    loss_cols = [
        "total_loss",
        "loss_cls",
        "loss_box_reg",
        "loss_mask",
        "loss_mask_point",
        "loss_rpn_cls",
        "loss_rpn_loc",
    ]

    plt.figure(figsize=(15, 10))
    for col in loss_cols:
        if (
            col in df.columns and not df[col].isnull().all()
        ):  # df[col].isnull().all() handles pd.NA
            plt.plot(df["iter"], df[col], label=col)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Training Losses vs. Iteration ({log_filename_prefix})")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(Path(output_dir) / f"{log_filename_prefix}_losses.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    if "lr" in df.columns and not df["lr"].isnull().all():
        plt.plot(df["iter"], df["lr"], label="Learning Rate")
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.title(f"Learning Rate vs. Iteration ({log_filename_prefix})")
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(output_dir) / f"{log_filename_prefix}_lr.png")
        plt.close()

    if "max_mem" in df.columns and not df["max_mem"].isnull().all():
        plt.figure(figsize=(10, 5))
        plt.plot(df["iter"], df["max_mem"], label="Max Memory (MB)")
        plt.xlabel("Iteration")
        plt.ylabel("Max Memory (MB)")
        plt.title(f"Max Memory vs. Iteration ({log_filename_prefix})")
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(output_dir) / f"{log_filename_prefix}_max_mem.png")
        plt.close()

    if (
        "time" in df.columns
        and "data_time" in df.columns
        and not df["time"].isnull().all()
        and not df["data_time"].isnull().all()
    ):
        plt.figure(figsize=(10, 5))
        plt.plot(df["iter"], df["time"], label="Total Time per Iter")
        plt.plot(df["iter"], df["data_time"], label="Data Time per Iter")
        plt.xlabel("Iteration")
        plt.ylabel("Time (s)")
        plt.title(f"Time per Iteration ({log_filename_prefix})")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            Path(output_dir) / f"{log_filename_prefix}_iteration_time.png"
        )
        plt.close()
    print(
        f"Training metrics plots saved to {output_dir} for"
        f" {log_filename_prefix}"
    )


def plot_evaluation_metrics(df, output_dir, log_filename_prefix):
    if df.empty:
        print(f"No evaluation data to plot for {log_filename_prefix}.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ap_metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

    for eval_type in ["bbox", "segm"]:
        eval_type_df = df[
            df["type"] == eval_type
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning
        if eval_type_df.empty:
            continue

        # Convert iter to numeric for plotting if it's not already
        eval_type_df["iter"] = pd.to_numeric(
            eval_type_df["iter"], errors="coerce"
        )
        eval_type_df.sort_values("iter", inplace=True)

        plt.figure(figsize=(15, 10))
        for metric in ap_metrics:
            if (
                metric in eval_type_df.columns
                and not eval_type_df[metric].isnull().all()
            ):
                # Ensure metric column is numeric
                eval_type_df[metric] = pd.to_numeric(
                    eval_type_df[metric], errors="coerce"
                )
                plt.plot(
                    eval_type_df["iter"],
                    eval_type_df[metric],
                    label=metric,
                    marker="o",
                )

        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.title(
            f"{eval_type.upper()} AP Metrics vs. Iteration"
            f" ({log_filename_prefix})"
        )
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.xticks(
            eval_type_df["iter"].unique()
        )  # Show all evaluation points on x-axis
        plt.savefig(
            Path(output_dir)
            / f"{log_filename_prefix}_{eval_type}_AP_metrics.png"
        )
        plt.close()
    print(
        f"Evaluation AP plots saved to {output_dir} for {log_filename_prefix}"
    )


def plot_per_category_ap_metrics(df, output_dir, log_filename_prefix):
    if df.empty:
        print(
            "No per-category evaluation data to plot for"
            f" {log_filename_prefix}."
        )
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    iterations = sorted(df["iter"].unique())

    for eval_type in ["bbox", "segm"]:
        type_df = df[df["type"] == eval_type]
        if type_df.empty:
            continue

        for iteration in iterations:
            iter_df = type_df[type_df["iter"] == iteration]
            if iter_df.empty:
                continue

            plt.figure(
                figsize=(10, 7)
            )  # Adjusted size for better category label visibility
            sns.barplot(
                x="category",
                y="AP",
                data=iter_df,
                palette="viridis",
                hue="category",
                dodge=False,
                legend=False,
            )  # Removed legend=False as it's deprecated and hue='category' implies it.
            plt.xlabel("Category")
            plt.ylabel("AP")
            plt.title(
                f"Per-Category {eval_type.upper()} AP at Iteration"
                f" {iteration} ({log_filename_prefix})"
            )
            plt.xticks(rotation=45, ha="right")
            plt.ylim(
                0, max(iter_df["AP"].max() * 1.1, 10)
            )  # Ensure y-axis starts at 0 and has some headroom
            plt.tight_layout()  # Adjust layout to prevent labels from overlapping
            plt.savefig(
                Path(output_dir)
                / f"{log_filename_prefix}_{eval_type}_per_category_AP_iter_{iteration}.png"
            )
            plt.close()
    print(
        f"Per-category AP plots saved to {output_dir} for"
        f" {log_filename_prefix}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse and plot metrics from Detectron2 log files."
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        help="Path to one or more log files (e.g., log1.txt log2.txt).",
    )
    parser.add_argument(
        "--output_dir_base",
        default="log_analysis_plots",
        help="Base directory to save plots.",
    )

    args = parser.parse_args()

    for log_file_path_str in args.log_files:
        log_file_path = Path(log_file_path_str)
        if not log_file_path.exists():
            print(f"Log file not found: {log_file_path}")
            continue

        print(f"Processing log file: {log_file_path}")
        log_filename_prefix = log_file_path.stem

        current_output_dir = Path(args.output_dir_base) / log_filename_prefix
        current_output_dir.mkdir(parents=True, exist_ok=True)

        training_df, evaluation_df, per_category_df = parse_log_file(
            log_file_path
        )

        print(f"\n--- Data for {log_filename_prefix} ---")
        if not training_df.empty:
            print(f"Found {len(training_df)} training log entries.")
        else:
            print("No training log entries found.")

        if not evaluation_df.empty:
            print(
                f"Found {len(evaluation_df)} overall evaluation log entries."
            )
        else:
            print("No overall evaluation log entries found.")

        if not per_category_df.empty:
            print(
                f"Found {len(per_category_df)} per-category evaluation log"
                " entries."
            )
        else:
            print("No per-category evaluation log entries found.")

        plot_training_metrics(
            training_df, current_output_dir, log_filename_prefix
        )
        plot_evaluation_metrics(
            evaluation_df, current_output_dir, log_filename_prefix
        )
        plot_per_category_ap_metrics(
            per_category_df, current_output_dir, log_filename_prefix
        )

        print(
            f"Finished processing {log_file_path}. Plots saved in"
            f" {current_output_dir}"
        )

    print(
        "\nReminder: Confusion matrices cannot be generated from these log"
        " files as they don't contain the necessary raw prediction data."
    )
    print(
        "If your log files have different loss names (e.g., for Cascade"
        " R-CNN), you may need to adjust the TRAIN_LOG_PATTERN in the script."
    )
