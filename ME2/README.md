# ME2 — Voice Controlled Smart Device

A tiny, **from-scratch**, **on-device** Voice Command Model (VCM) for a smart assistant:
always-on wake word → spoken-language understanding (intent + slot) → actuation, all on a
**Raspberry Pi 4** with **no cloud and no LLM**. Trained on the class's shared "gold" dataset
plus own-mic recordings; validated live on the Pi.

- **Model:** depthwise-separable CNN (SLUNet) — **22,951 params · 92 KB** fp32 ONNX
- **Gold-test accuracy:** intent **86.8%** · slot **96.7%** · **joint 84.3%**
- **On-Pi latency:** ~30–70 ms (onnxruntime, CPU) · RTF ≈ 0.02
- **Wake word:** "Hey Iris" — a separate tiny from-scratch keyword model
- **Everything on-device** — no cloud round-trip for recognition

## Pipeline

```
Mic · 16 kHz → log-mel 40×351 → DS-CNN backbone (width 64)
                                   ├── intent head (20 incl. OUT_OF_SCOPE)
                                   └── slot head  (19 incl. "none")
                → Raspberry Pi 4 actuators (LEDs · OLED SSD1306 · TM1637)
                → laptop companion (chime · voice narration · Spotify · phone via adb)
```

## Reproduce

### 1. Environment
```bash
conda create -n me2env python=3.11 -y && conda activate me2env
pip install -r requirements.txt           # training/dev (laptop or A100)
```

### 2. Train on the gold dataset (A100 / GPU)
```bash
python train_slu.py --hf-dataset /data/ai231 --noise ~/noise --music-as-noise ~/music \
    --aug-voice --label-smoothing 0.0 --arch dscnn --width 64 \
    --epochs 50 --batch 128 --workers 8 --run-name goldreal
```
Optional own-mic adaptation (record with `record_commands.py`, then `--real cmd_data --real-oversample 60`).

### 3. Benchmark + export
```bash
python benchmark_slu.py --ckpt checkpoints_slu/goldreal-best.ckpt --hf-dataset /data/ai231 --hf-split test --tag goldreal
python export_slu.py --ckpt checkpoints_slu/goldreal-best.ckpt     # -> slu.onnx
```

### 4. Run on the Raspberry Pi 4
See [`SETUP_PI.md`](SETUP_PI.md) for the full headless setup + wiring. Short version:
```bash
pip install -r requirements-rpi.txt        # onnxruntime + audio + GPIO, no torch
python pi_assistant.py --onnx slu.onnx --wake-word wake.onnx --student <id> \
    --oled --tm1637 20,21 --gpio --threshold 0.4 --wake-threshold 0.5 --device <mic-idx>
```
> The Pi inference path is **torch-free** — `features.py` (the NumPy front-end) replaces torchaudio,
> which won't run on the Pi 4's CPU. Validated to match the training features bit-for-bit.

### Wake word (optional, from scratch)
```bash
python record_wake.py --name Iris --pos 80 --neg 160        # on the Pi
python train_wake.py  --data wake_data --epochs 40          # on the laptop -> wake.onnx
```

### Laptop companion (chime · voice · Spotify · phone)
```bash
pip install pynput sounddevice numpy      # + Android platform-tools (adb) for phone
python companion.py --port 9009 --phone +63XXXXXXXXXX
```

## Repo guide

| Area | Files |
|---|---|
| Config / features | `config.py`, `features.py` (numpy), model `model.py` / `bcresnet.py` |
| Data loaders | `data.py` (gold HF + manifest + real recordings) |
| Train / eval / export | `train_slu.py`, `benchmark_slu.py`, `eval_reject.py`, `export_slu.py`, `qat.py` |
| Wake word | `record_wake.py`, `train_wake.py` |
| Own-mic adaptation | `record_commands.py`, `merge_data.py` |
| Raspberry Pi runtime | `pi_assistant.py`, `pi_gpio.py`, `pi_oled.py`, `pi_tm1637.py`, `pi_strip.py`, `pi_wake.py` |
| Laptop companion | `companion.py` |
| Dataset kit | `prompts.json`, `labelmap.json`, `record.py`, `validate_dataset.py`, `DATASET.md` |
| Results / plots | `metrics_*.json`, `*_metrics.csv`, `confusion_*.png`, `learning_curves.png`, `plot_curves.py`, `compare_runs.py` |

## Dataset

Class shared "gold" master set (collective work) on Hugging Face:
`airimonda/ai231-me2-voice-commands` (configs: `default`, `supplemental_synth`,
`synthetic_negatives`), plus own-mic recordings for fine-tuning. 20 intents (incl.
`OUT_OF_SCOPE`), 6 slotted intents (3 closed values each).

## Training environment

UP Diliman SHARC DGX — 1× NVIDIA A100-SXM4-40 GB (of 8), PyTorch Lightning, AdamW
(lr 3e-3, cosine), ~50 epochs, ~7 min wall-clock. Note: runs are **not seeded**.

## License

MIT — see [`LICENSE`](LICENSE).
