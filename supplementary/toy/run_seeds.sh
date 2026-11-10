#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
architecture="${1:?Usage: bash toy/run_seeds.sh transformer|hybrid [output-directory]}"
output_directory="${2:-outputs/toy}"
for seed in 101 102 103 104 105; do
  python toy/train.py --architecture "$architecture" --seed "$seed" --root "$output_directory" --device cuda
done
