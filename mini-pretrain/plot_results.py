import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
figures = root / "figures"
figures.mkdir(exist_ok=True)
background, ink, muted = "#f7f5f0", "#193139", "#687a7c"
colors = ["#a8553a", "#007c78", "#70609b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": ink,
                     "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
                     "figure.facecolor": background, "axes.facecolor": background,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False,
                     "svg.fonttype": "none", "savefig.facecolor": background})


def save(figure, name):
    for extension in ["png", "svg"]:
        figure.savefig(figures / f"{name}.{extension}", dpi=240, bbox_inches="tight", pad_inches=0.25)
    plt.close(figure)


archive = ZipFile(root / "logs.zip")
runs = []
for name in sorted(n for n in archive.namelist() if n.endswith("/metrics.json")):
    metric = json.loads(archive.read(name))
    history = [json.loads(line) for line in archive.read(name.replace("metrics.json", "history.jsonl")).decode().splitlines()]
    runs.append((Path(name).parent.name, metric, history))
assert runs, "No completed training runs"
labels = {name: f"{metric['settings']['learning_rate']:.0e} · {'cosine' if name == 'cosine' else 'constant'}" for name, metric, _ in runs}
figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.3), gridspec_kw={"width_ratios": [2.25, 1]}, layout="constrained")
figure.suptitle("15 минут с нуля", fontsize=27, fontweight="bold", x=0.04, ha="left")
axes[0].set_title("Qwen3 · русская Wikipedia · A100 80 GB", fontsize=12, color=muted, loc="left", pad=24)
for index, (name, metric, history) in enumerate(runs):
    points = [row for row in history if "loss" in row]
    times = np.array([row["elapsed_seconds"] / 60 for row in points])
    losses = np.array([row["loss"] for row in points])
    smooth = np.convolve(losses, np.ones(7) / 7, mode="valid")
    axes[0].plot(times, losses, color=colors[index], alpha=0.24, linewidth=0.8)
    axes[0].plot(times[6:], smooth, color=colors[index], linewidth=2, label=labels.get(name, name))
    value = metric["final"]["eval_loss"]
    axes[1].scatter(value, index, color=colors[index], s=90, zorder=3)
    axes[1].annotate(f"{value:.3f}", (value, index), xytext=(10, 0), textcoords="offset points", va="center", fontweight="bold")
axes[0].set(xlabel="Минуты обучения", ylabel="Train loss", xlim=(0, 15.1))
axes[0].set_xticks([0, 3, 6, 9, 12, 15])
axes[0].grid(axis="y", color="#dce2df", linewidth=0.7)
axes[0].legend(frameon=False, fontsize=10, loc="upper right")
axes[1].set_title("Финальный eval_loss", loc="left", fontweight="bold", pad=24)
axes[1].set_yticks(range(len(runs)), [labels.get(name, name) for name, _, _ in runs], fontsize=9)
axes[1].invert_yaxis()
axes[1].set_xlabel("Меньше лучше · 5000 статей")
values = [m["final"]["eval_loss"] for _, m, _ in runs]
spread = max(0.3, max(values) - min(values))
axes[1].set_xlim(min(values) - spread * 0.2, max(values) + spread * 0.7)
axes[1].grid(axis="x", color="#dce2df", linewidth=0.7)
figure.text(0.04, -0.025, "Тонкие линии: исходный loss. Толстые: среднее по 7 точкам. Один seed (42), по одному запуску на настройку.", fontsize=9, color=muted)
save(figure, "training")

probes = []
for name in sorted(n for n in archive.namelist() if n.startswith("probes/") and n.endswith(".json")):
    if name.endswith(".settings.json"):
        continue
    item = json.loads(archive.read(name))
    item["name"] = Path(name).stem
    probes.append(item)
throughput = [item for item in probes if item["status"] == "ok" and not item["name"].startswith("lr")]
throughput.sort(key=lambda item: np.median([r["token_slots"] / r["seconds"] for r in item["records"][3:]]))
figure, axis = plt.subplots(figsize=(12, 6.4), layout="constrained")
figure.suptitle("Скорость и память", fontsize=24, fontweight="bold", x=0.04, ha="left")
for index, item in enumerate(throughput):
    rates = np.array([row["token_slots"] / row["seconds"] / 1000 for row in item["records"][3:]])
    median = np.median(rates)
    q25, q75 = np.quantile(rates, [0.25, 0.75])
    axis.barh(index, median, height=0.55, color="#007c78" if item["name"] == "compile" else "#a5bfba", alpha=1.0)
    axis.errorbar(median, index, xerr=[[median - q25], [q75 - median]], color=ink, capsize=3, linewidth=1)
    axis.text(median + 0.6, index, f"{median:.1f} · {item['peak_memory_gb']:.1f} GB", va="center", fontsize=10)
probe_labels = {"b8": "Batch 8", "b16": "Batch 16", "b32": "Batch 32", "b48": "Batch 48",
                "b64": "Batch 64", "b16a2": "Batch 16 × accumulation 2", "unfused": "AdamW без fused",
                "compile": "torch.compile", "tf32off": "TF32 выключен", "fp16": "FP16, веса FP32"}
axis.set_yticks(range(len(throughput)), [probe_labels.get(item["name"], item["name"]) for item in throughput])
axis.set_xlabel("Тысячи позиций токенов в секунду, включая padding")
axis.set_xlim(0, max(np.median([r["token_slots"] / r["seconds"] / 1000 for r in p["records"][3:]]) for p in throughput) * 1.42)
axis.grid(axis="x", color="#dce2df", linewidth=0.7)
axis.set_axisbelow(True)
figure.text(0.04, -0.025, "По 12 шагов на пробу; первые 3 исключены. Медиана и квартили остальных 9. Справа: скорость и пиковая память.", fontsize=9, color=muted)
compiled_probe = next((p for p in throughput if p["name"] == "compile"), None)
cold_note = f" Первый шаг compile: {compiled_probe['records'][0]['seconds']:.1f} с." if compiled_probe else ""
figure.text(0.04, -0.065, "Batch 64: нехватка памяти. Остальные настройки как у batch 16." + cold_note, fontsize=9, color=muted)
save(figure, "throughput")

lr_probes = [item for item in probes if item["status"] == "ok" and item["name"].startswith("lr")]
figure, axis = plt.subplots(figsize=(11.5, 5.6), layout="constrained")
figure.suptitle("Выбор learning rate", fontsize=24, fontweight="bold", x=0.04, ha="left")
for index, item in enumerate(sorted(lr_probes, key=lambda p: p["settings"]["learning_rate"])):
    axis.plot([r["step"] for r in item["records"]], [r["loss"] for r in item["records"]],
              color=colors[index], linewidth=1.8,
              label=f"LR {item['settings']['learning_rate']:.0e} · eval {item['probe_eval_loss_128']:.3f}")
axis.set(xlabel="Шаги короткой пробы", ylabel="Train loss")
axis.grid(axis="y", color="#dce2df", linewidth=0.7)
axis.legend(frameon=False, loc="upper right")
figure.text(0.04, -0.025, "Одинаковые начальные веса и статьи; без warmup. Проба: 128 проверочных статей. Финальная оценка: все 5000.", fontsize=9, color=muted)
save(figure, "learning_rate")
