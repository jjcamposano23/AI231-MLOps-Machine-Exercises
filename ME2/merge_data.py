"""Merge Option B synthetic + real pooled recordings into ONE training manifest.

Real recordings come from the class pipeline (github.com/ayla011/ai231-me2-voice-data):
its master manifest has `label` (=intent), `slot_value`, `source_path` (wav path
relative to the raw/ dir), `speaker_id`, and whisper `status`/`wer`.

    # run this WHERE THE AUDIO LIVES (e.g. the DGX) so the absolute paths resolve:
    python merge_data.py \
        --optionb ~/AI231/MEX2/OptionB \
        --real ~/voice-data/master_manifest.csv --real-root ~/voice-data/raw \
        --out combined.csv --real-test-frac 0.2

Then retrain from scratch with noise:
    python train_slu.py --root . --manifest combined.csv --noise noise/ --epochs 60

Design: real voices define val/test (the honest, demo-like benchmark); ALL of the
synthetic data goes to train (broad coverage). Output columns match ManifestDS:
    path (ABSOLUTE), intent, slot, slot_value, speaker, split, source
"""
import argparse
import glob
import os
import random

import pandas as pd

from config import INTENTS


def _col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def load_optionb(folder, mode):
    """mode: 'train' (all synth -> train), 'keep_test' (synth TEST stays test, rest
    -> train), 'keep' (original Option B splits)."""
    df = pd.read_csv(os.path.join(folder, "manifest.csv"))
    orig = df["split"].astype(str).str.lower()
    if mode == "train":
        split = "train"
    elif mode == "keep_test":
        split = orig.map(lambda s: "test" if s == "test" else "train")
    else:
        split = orig
    return pd.DataFrame({
        "path":       df["path"].apply(lambda p: os.path.abspath(os.path.join(folder, str(p)))),
        "intent":     df["intent"].astype(str),
        "slot_value": df["slot_value"].fillna("").astype(str),
        "speaker":    "synth_" + df["speaker"].astype(str),
        "split":      split,
        "source":     "synthetic",
    })


def load_real(real_path, raw_root, wer_max):
    """`real_path` is EITHER a directory of speaker folders (each with a
    manifest.csv) OR a single master manifest.csv."""
    if os.path.isdir(real_path):                   # recordings/<speaker>/manifest.csv
        frames = []
        for man in sorted(glob.glob(os.path.join(real_path, "*", "manifest.csv"))):
            folder = os.path.dirname(man)
            d = pd.read_csv(man)
            d["_folder"] = folder
            d["_spk"] = os.path.basename(folder)
            frames.append(d)
        if not frames:
            raise SystemExit(f"no <speaker>/manifest.csv under {real_path}")
        df = pd.concat(frames, ignore_index=True)
    else:                                          # a single master manifest
        df = pd.read_csv(real_path)
        df["_folder"] = raw_root or os.path.dirname(os.path.abspath(real_path))
        df["_spk"] = None

    c_int = _col(df, ["intent", "label"])
    c_slot = _col(df, ["slot_value", "slot"])
    c_path = _col(df, ["source_path", "filename", "path", "file"])
    c_spk = _col(df, ["speaker_id", "speaker"])
    c_stat = _col(df, ["status"])
    c_wer = _col(df, ["wer", "word_error_rate"])
    if c_int is None or c_path is None:
        raise SystemExit(f"real manifest missing intent/path columns; has {list(df.columns)}")

    keep = pd.Series(True, index=df.index)
    if c_stat:                                     # keep only whisper-approved takes
        ok = {"approved", "ok", "pass", "accept", "accepted", "valid", "true", "kept"}
        keep &= df[c_stat].astype(str).str.lower().isin(ok)
    if c_wer and wer_max is not None:
        keep &= pd.to_numeric(df[c_wer], errors="coerce").fillna(1.0) <= wer_max
    df = df[keep]

    def speaker(row):
        return str(row["_spk"]) if row["_spk"] else (str(row[c_spk]) if c_spk else "unknown")

    out = pd.DataFrame({
        "path":       df.apply(lambda r: os.path.abspath(os.path.join(r["_folder"], str(r[c_path]))), axis=1),
        "intent":     df[c_int].astype(str),
        "slot_value": (df[c_slot].fillna("").astype(str) if c_slot else ""),
        "speaker":    "real_" + df.apply(speaker, axis=1),
        "split":      "",                          # assigned below
        "source":     "real",
    })
    print(f"real: kept {len(out)} approved takes from {out['speaker'].nunique()} speaker(s)")
    return out


def assign_real_splits(real, test_frac, val_frac, seed, min_speakers=3):
    """Speaker-independent hold-out when there are enough real speakers; otherwise
    (e.g. only your own voice) split BY UTTERANCE so the real voice is in TRAIN."""
    real = real.copy()
    rng = random.Random(seed)
    spks = sorted(real["speaker"].unique())
    if len(spks) >= min_speakers:
        # UNSEEN-SPEAKER test: hold out whole speaker(s); train on the rest; carve a
        # small utterance val slice from the training speakers (don't waste a speaker).
        rng.shuffle(spks)
        n_test = max(1, int(round(len(spks) * test_frac)))
        test_s = set(spks[:n_test])
        is_test = real["speaker"].isin(test_s)
        real.loc[is_test, "split"] = "test"
        rest = list(real.index[~is_test]); rng.shuffle(rest)
        n_val = max(1, int(len(rest) * val_frac))
        val_i = set(rest[:n_val])
        real.loc[~is_test, "split"] = [("val" if i in val_i else "train")
                                       for i in real.index[~is_test]]
        print(f"real split: SPEAKER-HELD-OUT — {n_test} of {len(spks)} speaker(s) unseen in "
              f"test (honest 'any speaker' estimate); train on the rest, val = utterance slice")
    else:
        idx = list(real.index); random.Random(seed).shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * test_frac)); n_val = max(1, int(n * val_frac))
        test_i, val_i = set(idx[:n_test]), set(idx[n_test:n_test + n_val])
        real["split"] = [("test" if i in test_i else "val" if i in val_i else "train")
                         for i in real.index]
        print(f"real split: BY UTTERANCE ({len(spks)} speaker(s)) — most in TRAIN so the "
              f"model learns your voice; test = held-out takes of the SAME voice "
              f"(own-voice estimate, not speaker-independent)")
    return real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optionb", required=True, help="Option B folder (has manifest.csv)")
    ap.add_argument("--real", default=None, help="real master_manifest.csv (optional)")
    ap.add_argument("--real-root", default=None, help="raw/ dir the source_path is relative to")
    ap.add_argument("--out", default="combined.csv")
    ap.add_argument("--real-test-frac", type=float, default=0.2, help="real speakers held out for test")
    ap.add_argument("--real-val-frac", type=float, default=0.1)
    ap.add_argument("--wer-max", type=float, default=0.5, help="drop real takes above this WER")
    ap.add_argument("--keep-synth-test", action="store_true",
                    help="keep Option B's synthetic TEST as a held-out test (source=synthetic)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    ob_mode = "keep" if not args.real else ("keep_test" if args.keep_synth_test else "train")
    parts = [load_optionb(args.optionb, ob_mode)]
    if args.real:
        raw_root = args.real_root or os.path.dirname(os.path.abspath(args.real))
        real = load_real(args.real, raw_root, args.wer_max)
        parts.append(assign_real_splits(real, args.real_test_frac, args.real_val_frac, args.seed))
    else:
        print("[note] no --real given: writing Option B only (its original splits).")

    comb = pd.concat(parts, ignore_index=True)

    bad = sorted(set(comb["intent"]) - set(INTENTS))
    if bad:
        print(f"[warn] dropping rows with intents not in config: {bad}")
        comb = comb[comb["intent"].isin(INTENTS)]
    comb["slot"] = comb["slot_value"]

    comb.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(comb)} rows)")
    print("by split :", comb["split"].value_counts().to_dict())
    print("by source:", comb["source"].value_counts().to_dict())
    print("val/test speakers are REAL voices — the honest, demo-like benchmark.")


if __name__ == "__main__":
    main()
