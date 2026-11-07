#!/usr/bin/env bash
# Pi side of King's fd-benchmark: run Iris in benchmark mode (no mic/speaker; audio comes from the laptop's fdbench).
#   bash ~/ME3/scripts/bench_pi.sh            (stop the normal "python -m assistant.main" first: Ctrl-C)
set -e
cd "$(dirname "$0")/.."
LAPTOP_IP="${LAPTOP_IP:-192.168.1.3}"
echo "FDBENCH_URL=ws://${LAPTOP_IP}:8791" > fd-benchmark/.env
.venv/bin/python fd-benchmark/launch_assistant.py --check
exec .venv/bin/python fd-benchmark/launch_assistant.py
