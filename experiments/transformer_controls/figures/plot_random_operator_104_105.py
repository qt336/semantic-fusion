"""Plot actual test evaluations from the ongoing random-output controls."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "history_curves_104_105"
OUT.mkdir(exist_ok=True)
font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "pdf.fonttype": 42})

snapshots = {}
exports = []
manifest = {"created": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "sources": {}}
for seed in (104, 105):
    source = ROOT / f"runs/random_operator_seed{seed}/history.jsonl"
    raw = source.read_bytes()
    # A live writer may have started, but not finished, the last JSON line.
    complete = raw[:raw.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    evaluated = [r for r in rows if r.get("evaluation")]
    assert evaluated and len({r["epoch"] for r in evaluated}) == len(evaluated)
    snapshots[seed] = evaluated
    manifest["sources"][str(seed)] = {
        "path": str(source), "complete_history_sha256": hashlib.sha256(complete).hexdigest(),
        "trained_epoch": rows[-1]["epoch"], "last_test_epoch": evaluated[-1]["epoch"],
        "test_points": len(evaluated),
    }
    for row in evaluated:
        for condition, metrics in row["evaluation"].items():
            n, counts = metrics["n"], metrics["counts"]
            assert sum(counts.values()) == n
            record = {"seed": seed, "epoch": row["epoch"], "condition": condition, "n": n}
            for outcome in ("correct", "first", "query", "eos", "other"):
                count = counts.get(outcome, 0)
                record[f"{outcome}_count"] = count
                record[f"{outcome}_percent"] = 100 * count / n
                assert abs(metrics["rates"].get(outcome, 0) - count / n) < 1e-9
            exports.append(record)

with (OUT / "test_curves.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(exports[0]))
    writer.writeheader()
    writer.writerows(exports)

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
max_epoch = max(rows[-1]["epoch"] for rows in snapshots.values())
for column, (seed, rows) in enumerate(snapshots.items()):
    x = [r["epoch"] for r in rows]
    top, bottom = axes[:, column]
    accuracy_series = [
        ("ordinary_9_10", "9,10 留出检索", "#176baf", "-", 2.6),
        ("arithmetic", "真实算术", "#d48317", "--", 1.5),
        ("standard", "普通检索", "#25826b", ":", 1.8),
    ]
    for condition, label, color, style, width in accuracy_series:
        y = [100 * r["evaluation"][condition]["accuracy"] for r in rows]
        top.plot(x, y, color=color, linestyle=style, linewidth=width,
                 marker="o", markersize=3, label=label)
        if condition == "ordinary_9_10":
            top.annotate(f"{y[-1]:.2f}%", (x[-1], y[-1]), xytext=(-7, 10),
                         textcoords="offset points", ha="right", color=color,
                         fontsize=11, fontweight="bold")
    top.set_title(f"Seed {seed} · 测试正确率（最新第 {x[-1]} 轮）", fontsize=12)
    top.set_ylabel("正确率（%）")
    for outcome, label, color in [
        ("first", "错答首 token：9", "#bd3d48"),
        ("eos", "错答 EOS", "#855ba6"),
        ("other", "其他错误（含查询 token）", "#808998"),
    ]:
        y = []
        for r in rows:
            m = r["evaluation"]["ordinary_9_10"]
            count = m["counts"][outcome]
            if outcome == "other":
                count += m["counts"].get("query", 0)
            y.append(100 * count / m["n"])
        bottom.plot(x, y, color=color, linewidth=1.9, marker="o", markersize=3, label=label)
    bottom.set_title(f"Seed {seed} · 9,10 检索错误分布", fontsize=12)
    bottom.set_ylabel("占全部测试题的比例（%）")
    bottom.set_xlabel("训练轮数（epoch）")
    for ax in (top, bottom):
        ax.set_ylim(-2, 106)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlim(0, max_epoch + max_epoch * .025)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9)

fig.suptitle("随机输出 9,10 对照：测试表现变化（训练中）", fontsize=18, y=.985)
fig.text(.5, .935, "9,10 每轮出现 18 万次，检索训练中留出；真实算符各出现 9 万次", ha="center", color="#4b5563")
fig.text(.075, .035,
         "第 1 轮及每 25 轮评估；折线连接实际测试点，未平滑、未插补。\n"
         "9,10 每次使用固定 2,560 题；下排各类错误均以全部题目为分母，与正确率合计 100%。",
         fontsize=10, color="#4b5563")
fig.subplots_adjust(left=.075, right=.98, bottom=.13, top=.87, hspace=.33, wspace=.17)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"test_accuracy_curves.{ext}", dpi=180, facecolor="white")
plt.close(fig)

lines = ["# 随机输出 9,10：104、105 测试曲线", "", f"生成时间：{manifest['created']}", "",
         "这是一份生成时的快照。训练继续进行；运行 `python figures/plot_random_operator_104_105.py` 可更新曲线。", "",
         "![测试曲线](test_accuracy_curves.png)", "",
         "[PDF](test_accuracy_curves.pdf) · [逐次测试数据 CSV](test_curves.csv)", "",
         "| Seed | 最新测试轮数 | 9,10 正确率 | 错答 9 | EOS | 真实算术 | 普通检索 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for seed, rows in snapshots.items():
    e = rows[-1]["evaluation"]
    r = e["ordinary_9_10"]["rates"]
    values = [r["correct"], r["first"], r["eos"], e["arithmetic"]["accuracy"], e["standard"]["accuracy"]]
    lines.append(f"| {seed} | {rows[-1]['epoch']} | " + " | ".join(f"{v * 100:.2f}%" for v in values) + " |")
lines += ["", "所有曲线使用保存的原始评估计数。测试点之间的行为未作估计，不以曲线峰值替代固定终点结果。", ""]
(OUT / "results.md").write_text("\n".join(lines))
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(OUT / "test_accuracy_curves.png")
