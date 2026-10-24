#!/usr/bin/env bash
# Run from an activated environment with requirements-dev.txt installed.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
python train_net.py --seed 1 --output runs/ppo
python train_net.py --seed 2 --output runs/ppo_seed2
python train_net.py --seed 3 --output runs/ppo_seed3
mkdir -p models results
cp runs/ppo/best.pt models/ppo.pt
cp runs/ppo_seed2/best.pt models/ppo_seed2.pt
cp runs/ppo_seed3/best.pt models/ppo_seed3.pt
for seed in 1 2 3; do
    run="runs/ppo_seed${seed}"
    if [[ "$seed" == 1 ]]; then run="runs/ppo"; fi
    python evaluate.py --model "$run/best.pt" --output "results/ppo_seed${seed}.json"
    cp "$run/metrics.csv" "results/training_seed${seed}.csv"
    cp "$run/config.json" "results/config_seed${seed}.json"
done
for baseline in random coast heuristic; do
    python evaluate.py --baseline "$baseline" --output "results/${baseline}.json"
done
python evaluate.py --model models/ppo.pt --stochastic --output results/ppo_seed1_sampled.json
python evaluate.py --model models/ppo.pt --max-steps 30000 --output results/ppo_seed1_long.json
python evaluate.py --baseline heuristic --max-steps 30000 --output results/heuristic_long.json
python scripts/report.py
