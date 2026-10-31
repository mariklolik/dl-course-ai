import argparse
import csv
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("experiments.json").read_text())
    archive_path = args.output / "raw_results.zip"
    archive = zipfile.ZipFile(archive_path) if archive_path.exists() else None

    def read(name: str, filename: str):
        path = args.output / name / filename
        content = path.read_text() if path.exists() else archive.read(f"{name}/{filename}").decode()
        return json.loads(content)

    rows = []
    for name in config["experiments"]:
        result = read(name, "summary.json")
        training, metrics = result["training"], result["metrics"]
        case = result["case"]
        row = {
            "case": name, "r": case.get("r", ""), "alpha": case.get("alpha", ""),
            "dropout": case.get("dropout", ""), "target_modules": json.dumps(case.get("target_modules", [])),
            "trainable_parameters": training["trainable_parameters"], "total_parameters": training["total_parameters"],
            "train_seconds": training["train_seconds"], "peak_allocated_gib": training["peak_allocated_gib"],
            "peak_reserved_gib": training["peak_reserved_gib"], "optimizer_steps": training["optimizer_steps"],
        }
        for split in ("validation", "test"):
            for key in ("n", "accuracy", "macro_f1", "invalid_outputs", "completion_nll"):
                row[f"{split}_{key}"] = metrics[split][key]
        rows.append(row)
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160})
    names = [row["case"] for row in rows]
    positions = list(range(len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
    axes[0].barh([x - 0.18 for x in positions], [100 * r["test_accuracy"] for r in rows], height=0.35, color="#1e5a78", label="Accuracy")
    axes[0].barh([x + 0.18 for x in positions], [100 * r["test_macro_f1"] for r in rows], height=0.35, color="#d77a3f", label="Macro-F1")
    axes[0].set(yticks=positions, yticklabels=names, xlabel="Test score, %", title="Same 2,000 test examples", xlim=(0, 100))
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper right")
    axes[1].barh([x - 0.18 for x in positions], [r["peak_allocated_gib"] for r in rows], height=0.35, color="#1e5a78", label="Allocated")
    axes[1].barh([x + 0.18 for x in positions], [r["peak_reserved_gib"] for r in rows], height=0.35, color="#d77a3f", label="Reserved")
    for index, row in enumerate(rows):
        if row["case"] != "base":
            axes[1].text(row["peak_reserved_gib"] + 0.4, index, f"{row['train_seconds']:.1f}s", va="center", fontsize=8)
    axes[1].set(yticks=positions, yticklabels=names, xlabel="Peak training GPU memory, GiB", title="Annotations: training wall time", xlim=(0, max(r["peak_reserved_gib"] for r in rows) + 8))
    axes[1].invert_yaxis()
    axes[1].legend(loc="upper right")
    fig.savefig(args.output / "comparison.png")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for name, color in [("full", "#1e5a78"), ("lora", "#d77a3f")]:
        history = [row for row in read(name, "training_history.json") if "loss" in row]
        axis.plot([row["step"] for row in history], [row["loss"] for row in history], label=name, color=color, linewidth=1.8)
    axis.set(xlabel="Optimizer step", ylabel="Completion-only training NLL", title="One epoch, effective batch size 32")
    axis.legend()
    axis.grid(alpha=0.15)
    fig.savefig(args.output / "training.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
