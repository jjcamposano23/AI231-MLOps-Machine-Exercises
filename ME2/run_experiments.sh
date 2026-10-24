#!/usr/bin/env bash
# ME2 — A/B/C/D experiment driver (run on the DGX, in ~/ME2 with me2env active).
#
#   bash run_experiments.sh combined.csv ~/noise ~/music
#
# All 4 runs share the SAME test split (from the manifest), so results compare fairly.
#   A = synthetic only
#   B = synthetic + your voice
#   C = B + noise augmentation
#   D = C + music + voice-aug (pitch/speed) + intent-only class-weight (optimized for any-speaker)
set -e

MANIFEST="${1:-combined.csv}"
NOISE="${2:-$HOME/noise}"
MUSIC="${3:-$HOME/music}"
EPOCHS="${EPOCHS:-60}"
COMMON="--root . --manifest $MANIFEST --arch dscnn --width 64 --epochs $EPOCHS --batch 256 --workers 8"

echo "=== Run A: synthetic only ==="
python train_slu.py $COMMON --train-source synthetic --run-name A
python benchmark_slu.py --root . --manifest "$MANIFEST" --ckpt checkpoints_slu/A-best.ckpt --tag A --noise "$NOISE"

echo "=== Run B: synthetic + voice ==="
python train_slu.py $COMMON --train-source all --run-name B
python benchmark_slu.py --root . --manifest "$MANIFEST" --ckpt checkpoints_slu/B-best.ckpt --tag B --noise "$NOISE"

echo "=== Run C: + noise ==="
python train_slu.py $COMMON --train-source all --noise "$NOISE" --run-name C
python benchmark_slu.py --root . --manifest "$MANIFEST" --ckpt checkpoints_slu/C-best.ckpt --tag C --noise "$NOISE"

echo "=== Run D: + music + voice-aug + intent-only class-weight (optimized for any-speaker) ==="
python train_slu.py $COMMON --train-source all --noise "$NOISE" --music-as-noise "$MUSIC" --aug-voice --class-weight-intent-only --run-name D
python benchmark_slu.py --root . --manifest "$MANIFEST" --ckpt checkpoints_slu/D-best.ckpt --tag D --noise "$NOISE"

echo "=== Compare ==="
python compare_runs.py --runs A B C D
echo "Done. See comparison.csv, comparison_bars.png, validation_curves.png, confusion_grid.png"
