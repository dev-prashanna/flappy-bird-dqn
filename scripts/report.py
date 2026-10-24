"""Rebuild figures, statistics, and the paper from recorded experiment files."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT / "results"
    output = ROOT / "paper" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    names = ["coast", "random", "heuristic", "ppo_seed1", "ppo_seed2", "ppo_seed3", "ppo_seed1_sampled"]
    reports = {name: json.loads((results / f"{name}.json").read_text()) for name in names}
    rng = np.random.default_rng(2026)
    summary = {}
    for name, report in reports.items():
        scores = np.asarray([r["score"] for r in report["results"]])
        boot = rng.choice(scores, (10000, len(scores)), replace=True).mean(1)
        summary[name] = {k: report[k] for k in ["mean_score", "std_score", "median_score", "mean_length", "survival_rate"]}
        summary[name]["bootstrap_score_ci95"] = np.quantile(boot, [0.025, 0.975]).tolist()
    seed_means = np.asarray([reports[f"ppo_seed{i}"]["mean_score"] for i in [1, 2, 3]])
    summary["across_training_seeds"] = {"mean_score": float(seed_means.mean()), "std_score": float(seed_means.std(ddof=1)), "n": 3}
    # Paired bootstrap keeps the pipe-seed pairing for PPO versus heuristic.
    difference = np.asarray([r["score"] for r in reports["ppo_seed1"]["results"]]) - np.asarray([r["score"] for r in reports["heuristic"]["results"]])
    boot = rng.choice(difference, (10000, len(difference)), replace=True).mean(1)
    summary["ppo1_minus_heuristic"] = {"mean_difference": float(difference.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}
    for name in ["ppo_seed1_long", "heuristic_long"]:
        report = json.loads((results / f"{name}.json").read_text())
        summary[name] = {k: report[k] for k in ["mean_score", "std_score", "mean_length", "survival_rate", "min_score", "max_score"]}
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    manifest = {}
    for seed in [1, 2, 3]:
        with (results / f"training_seed{seed}.csv").open() as file:
            rows = list(csv.DictReader(file))
        model_path = ROOT / "models" / ("ppo.pt" if seed == 1 else f"ppo_seed{seed}.pt")
        if model_path.exists():
            step = reports[f"ppo_seed{seed}"]["training_steps"]
            selected = next(r for r in rows if int(r["global_step"]) == step)
            manifest[str(model_path.relative_to(ROOT))] = {
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "seed": seed,
                "global_step": step, "validation_mean_score": float(selected["validation_score"])}
        axes[0].plot([int(r["global_step"]) for r in rows], [float(r["mean_score_100"]) for r in rows], label=f"Seed {seed}")
        valid = [r for r in rows if r["validation_score"] != ""]
        axes[1].plot([int(r["global_step"]) for r in valid], [float(r["validation_score"]) for r in valid], marker=".", label=f"Seed {seed}")
    (results / "checkpoints.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for ax in axes:
        ax.set_xlabel("Training physics frames")
        ax.set_ylabel("Mean pipes passed")
        ax.grid(alpha=0.2)
        ax.legend()
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set_title("Last 100 completed training episodes (sampled)")
    axes[1].set_title("Fixed validation courses (greedy)")
    fig.savefig(output / "learning_curves.pdf")
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)
    labels = ["Coast", "Random", "Heuristic", "PPO 1", "PPO 2", "PPO 3", "PPO 1\nsampled"]
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    means = [reports[n]["mean_score"] for n in names]
    ax.bar(labels, means, color=["#9babb9"]*2+["#d68b38"]+["#238b8e"]*4)
    for x, name in enumerate(names):
        scores = [r["score"] for r in reports[name]["results"]]
        jitter = np.linspace(-0.17, 0.17, len(scores))
        ax.scatter(x+jitter, scores, s=10, alpha=0.6, color="#172b4d")
    ax.axhline(68, color="gray", linestyle="--", linewidth=1, label="6,000-frame ceiling")
    ax.set_ylabel("Pipes passed")
    ax.set_title("Thirty held-out pipe courses per policy; dots are individual episodes")
    ax.legend(loc="lower right")
    fig.savefig(output / "evaluation.pdf")
    fig.savefig(output / "evaluation.png", dpi=180)
    plt.close(fig)
    table = ["| Policy | Mean score | SD | Median | Mean frames | Reached time limit |", "|:--|--:|--:|--:|--:|--:|"]
    for label, name in zip(["Coast", "Uniform random", "Height heuristic", "PPO seed 1 (greedy)", "PPO seed 2 (greedy)", "PPO seed 3 (greedy)", "PPO seed 1 (sampled)"], names):
        r = reports[name]
        table.append(f"| {label} | {r['mean_score']:.2f} | {r['std_score']:.2f} | {r['median_score']:.1f} | {r['mean_length']:.1f} | {100*r['survival_rate']:.1f}% |")
    long_table = ["| Controller | Mean pipes | SD | Mean frames | Reached time limit |",
                  "|:--|--:|--:|--:|--:|"]
    for label, name in [("Reference PPO (greedy)", "ppo_seed1_long"), ("Height heuristic", "heuristic_long")]:
        r = summary[name]
        long_table.append(f"| {label} | {r['mean_score']:.2f} | {r['std_score']:.2f} | {r['mean_length']:.1f} | {100*r['survival_rate']:.1f}% |")
    template = (ROOT / "paper" / "paper.template.md").read_text()
    replacements = {"RESULTS_TABLE": "\n".join(table), "LONG_RESULTS_TABLE": "\n".join(long_table),
                    "SEED1_MEAN": f"{reports['ppo_seed1']['mean_score']:.2f}",
                    "SEED1_SURVIVAL": f"{100*reports['ppo_seed1']['survival_rate']:.1f}",
                    "ACROSS_MEAN": f"{seed_means.mean():.2f}",
                    "ACROSS_SD": f"{seed_means.std(ddof=1):.2f}",
                    "PAIRED_MEAN": f"{difference.mean():.2f}",
                    "PAIRED_LOW": f"{summary['ppo1_minus_heuristic']['ci95'][0]:.2f}",
                    "PAIRED_HIGH": f"{summary['ppo1_minus_heuristic']['ci95'][1]:.2f}"}
    for key, value in replacements.items():
        template = template.replace("{{"+key+"}}", value)
    (ROOT / "paper" / "paper.md").write_text(template)
    print("Rebuilt results/summary.json, figures, and paper/paper.md")


if __name__ == "__main__":
    main()
