#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper"
pandoc paper.md --standalone --resource-path=. -o paper.tex
tectonic --keep-logs paper.tex
