#!/usr/bin/env bash
# One-time setup on the Raspberry Pi (64-bit Pi OS, Pi 4/5). Run from the ME3 folder:  bash scripts/install_pi.sh
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev portaudio19-dev libportaudio2 libsndfile1 mpv \
    python3-lgpio i2c-tools alsa-utils

# venv with system site-packages so the apt lgpio (TM1637) is visible
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
# optional actuator drivers from ME2 (strip needs sudo at runtime; see README)
.venv/bin/pip install rpi_ws281x luma.oled || echo "[warn] actuator drivers not installed (home tools will simulate)"

[ -f .env ] || cp .env.example .env
echo
echo "Done. Next:"
echo "  1) put your keys in .env (DEEPGRAM_API_KEY, OPENROUTER_API_KEY, TAVILY_API_KEY)"
echo "  2) .venv/bin/python -m assistant.main --list-devices   -> set INPUT_DEVICE_INDEX / OUTPUT_DEVICE_INDEX"
echo "  3) .venv/bin/python -m assistant.main"
