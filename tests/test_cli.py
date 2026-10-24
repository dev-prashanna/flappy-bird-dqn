"""A real collection/update/evaluation/save/load/continuation workflow."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                          check=True, capture_output=True, text=True, timeout=60)


def test_train_evaluate_and_continue(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    settings = ["--total-timesteps", "64", "--num-envs", "2", "--num-steps", "16",
                "--epochs", "1", "--minibatch-size", "16", "--eval-every", "1", "--eval-episodes", "2"]
    run("train_net.py", *settings, "--output", first)
    assert all((first / name).is_file() for name in ["best.pt", "latest.pt", "final.pt", "metrics.csv", "config.json"])
    output = tmp_path / "evaluation.json"
    run("evaluate.py", "--model", first / "best.pt", "--episodes", "2", "--max-steps", "20", "--output", output)
    report = json.loads(output.read_text())
    assert report["episodes"] == 2 and len(report["results"]) == 2
    assert report["mean_length"] <= 20
    run("train_net.py", *settings, "--resume", first / "final.pt", "--output", second)
    import torch
    checkpoint = torch.load(second / "final.pt", weights_only=True)
    assert checkpoint["global_step"] == 128
