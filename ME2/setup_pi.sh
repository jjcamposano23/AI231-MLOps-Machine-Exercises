#!/usr/bin/env bash
# One-shot VCM setup for a fresh Raspberry Pi OS (64-bit, Bookworm).
# Copy this + the 6 VCM files to the Pi, then:  bash setup_pi.sh
set -e

echo "== 1/4  system packages =="
sudo apt update
sudo apt install -y python3-venv python3-dev portaudio19-dev libportaudio2

echo "== 2/4  python venv (~/vcm) + deps =="
python3 -m venv ~/vcm
source ~/vcm/bin/activate
pip install --upgrade pip
pip install numpy onnxruntime soundfile sounddevice webrtcvad-wheels gpiozero lgpio pyserial

echo "== 3/4  audio devices (find your USB mic's index) =="
python - <<'PY'
import sounddevice as sd
print(sd.query_devices())
print("default input:", sd.default.device)
PY

echo "== 4/4  sanity: ONNX model loads =="
if [ -f "$HOME/slu.onnx" ]; then
  python - <<'PY'
import onnxruntime as ort
s = ort.InferenceSession("slu.onnx", providers=["CPUExecutionProvider"])
print("slu.onnx OK | inputs", [i.name for i in s.get_inputs()],
      "| outputs", [o.name for o in s.get_outputs()])
PY
else
  echo "   (slu.onnx not in \$HOME yet -- copy the 6 VCM files here)"
fi

echo
echo "== done =="
echo "Activate the env in every new shell with:   source ~/vcm/bin/activate"
echo "Then test recognition:"
echo "  python pi_assistant.py --onnx slu.onnx --student <id> --no-wake --threshold 0.4"
