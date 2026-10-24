"""Dataset, augmentation, and speaker-aware splitting.

Expected layout (one folder per class, folder name must match config.LABELS):

    data/
        play_music/     spk01_0001.wav  spk02_0007.wav ...
        query_search/   ...
        lights_onoff/   ...
        ...
        _unknown_/      random non-command speech
        _silence_/      background-noise-only clips

Filenames should start with a speaker id followed by '_', e.g. "spk01_0003.wav",
so the split can keep each speaker entirely in one of train/val/test
(speaker-independent evaluation -- required for an honest benchmark).
"""
import glob
import os
import random

import torch
import torchaudio
from torch.utils.data import DataLoader

from config import LABELS, LABEL2IDX, SR, CLIP_LEN, INTENT2IDX, slot_index, OOS
from features import logmel, pad_or_trim


# ------------------------------------------------------------------ manifest ---
def scan_dataset(root):
    """Return a list of (wav_path, label_idx) for every clip under `root`."""
    items = []
    for name in LABELS:
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            print(f"[warn] missing class folder: {folder}")
            continue
        for path in glob.glob(os.path.join(folder, "*.wav")):
            items.append((path, LABEL2IDX[name]))
    if not items:
        raise RuntimeError(f"No .wav files found under {root!r}")
    return items


def _speaker_of(path):
    return os.path.basename(path).split("_")[0]


def make_splits(items, val_frac=0.1, test_frac=0.1, seed=1337, by_speaker=True):
    """Split items into (train, val, test).

    If by_speaker, whole speakers are assigned to a single split so no speaker
    appears in more than one -- this prevents optimistic, leaky accuracy.
    """
    rng = random.Random(seed)
    if by_speaker:
        speakers = sorted({_speaker_of(p) for p, _ in items})
        rng.shuffle(speakers)
        n_test = max(1, int(len(speakers) * test_frac))
        n_val = max(1, int(len(speakers) * val_frac))
        test_spk = set(speakers[:n_test])
        val_spk = set(speakers[n_test:n_test + n_val])
        train, val, test = [], [], []
        for p, y in items:
            s = _speaker_of(p)
            (test if s in test_spk else val if s in val_spk else train).append((p, y))
        return train, val, test
    # fallback: plain random split
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    return shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val], shuffled[:n_test]


# ------------------------------------------------------------------- dataset ---
class CommandDS(torch.utils.data.Dataset):
    def __init__(self, items, noise_dir=None, train=True):
        self.items = items
        self.noises = glob.glob(os.path.join(noise_dir, "*.wav")) if noise_dir else []
        self.train = train
        self.tmask = torchaudio.transforms.TimeMasking(time_mask_param=25)
        self.fmask = torchaudio.transforms.FrequencyMasking(freq_mask_param=7)

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return wav.mean(0, keepdim=True)            # -> mono (1, T)

    def _random_noise(self):
        n, sr = torchaudio.load(random.choice(self.noises))
        if sr != SR:
            n = torchaudio.functional.resample(n, sr, SR)
        n = n.mean(0, keepdim=True)
        return pad_or_trim(n)

    def __getitem__(self, i):
        path, y = self.items[i]
        wav = pad_or_trim(self._load(path))

        if self.train:
            wav = torch.roll(wav, random.randint(-1600, 1600), dims=-1)   # time shift
            if self.noises and random.random() < 0.7:                     # add noise
                noise = self._random_noise()
                snr_db = random.uniform(5.0, 20.0)
                scale = (wav.norm() / (noise.norm() + 1e-6)) * (10 ** (-snr_db / 20))
                wav = wav + noise * scale

        feat = logmel(wav)                          # (1, N_MELS, N_FRAMES)
        if self.train:
            feat = self.fmask(self.tmask(feat))     # SpecAugment
        return feat, y


# ------------------------------------------------------------------- loaders ---
def build_loaders(data_root, noise=None, batch=128, workers=4, seed=1337):
    """Return (train_loader, val_loader, test_items, counts) with a
    speaker-independent split shared by every script (same seed)."""
    items = scan_dataset(data_root)
    tr_items, va_items, te_items = make_splits(items, seed=seed, by_speaker=True)
    tr = DataLoader(CommandDS(tr_items, noise, train=True), batch_size=batch,
                    shuffle=True, num_workers=workers,
                    persistent_workers=workers > 0)
    va = DataLoader(CommandDS(va_items, None, train=False), batch_size=batch,
                    num_workers=workers, persistent_workers=workers > 0)
    return tr, va, te_items, (len(tr_items), len(va_items), len(te_items))


# ============================================================================ #
#  Option B: manifest-driven SLU loading (intent + slot labels)                #
# ============================================================================ #
class ManifestDS(torch.utils.data.Dataset):
    """Reads rows of Option B's manifest.csv -> (log-mel, intent_idx, slot_idx).

    `root` is the OptionB/ folder; each row's `path` is relative to it.
    """

    def __init__(self, root, rows, train=False, noise_dir=None, music_dir=None, aug_voice=False):
        self.root = root
        self.rows = rows.reset_index(drop=True)
        self.train = train
        self.aug_voice = aug_voice
        self.noises = _clip_bank(noise_dir) if (train and noise_dir) else []
        self.music = _clip_bank(music_dir) if (train and music_dir) else []
        self.tmask = torchaudio.transforms.TimeMasking(time_mask_param=25)
        self.fmask = torchaudio.transforms.FrequencyMasking(freq_mask_param=7)

    def __len__(self):
        return len(self.rows)

    def _load(self, path, relative=True):
        p = os.path.join(self.root, path) if relative else path
        wav, sr = torchaudio.load(p)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return wav.mean(0, keepdim=True)                 # mono (1, T)

    def _random_clip(self, pool):
        return pool[random.randrange(len(pool))]         # pool = pre-cached (1, CLIP_LEN) clips

    def _mix(self, wav, bg, snr_lo, snr_hi):
        snr = random.uniform(snr_lo, snr_hi)
        scale = (wav.norm() / (bg.norm() + 1e-6)) * (10 ** (-snr / 20))
        return wav + bg * scale

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        wav = self._load(r["path"])                              # raw (pad after voice-aug)
        if self.train and self.aug_voice and random.random() < 0.5:
            wav = torchaudio.functional.resample(                # cheap GCD-friendly speed/pitch (10:9 / 10:11)
                wav, SR, random.choice([SR * 9 // 10, SR * 11 // 10]))
        wav = pad_or_trim(wav)
        if self.train:
            wav = torch.roll(wav, random.randint(-1600, 1600), dims=-1)
            if len(self.noises) and random.random() < 0.6:       # ambient noise, varied "distance"
                wav = self._mix(wav, self._random_clip(self.noises), 5.0, 20.0)
            if len(self.music) and random.random() < 0.3:        # music playing in background (quieter)
                wav = self._mix(wav, self._random_clip(self.music), 8.0, 25.0)
        feat = logmel(wav)
        if self.train:
            feat = self.fmask(self.tmask(feat))          # SpecAugment
        return feat, INTENT2IDX[r["intent"]], slot_index(r.get("slot_value"))


def build_loaders_manifest(root, manifest=None, batch=128, workers=4, noise=None,
                           music=None, train_source="all", aug_voice=False):
    """Split by the manifest `split` column -> (train_loader, val_loader, test_df, counts).

    `noise`/`music` = folders of .wav mixed into TRAIN clips (see ManifestDS).
    `train_source` in {all, synthetic, real} filters TRAIN rows by the `source`
    column (val/test are never filtered, so all runs share the same eval set).
    """
    import pandas as pd
    manifest = manifest or os.path.join(root, "manifest.csv")
    df = pd.read_csv(manifest)
    sp = df["split"].astype(str).str.lower()
    tr_df = df[sp.isin(["train"])]
    if train_source != "all" and "source" in df.columns:
        tr_df = tr_df[tr_df["source"].astype(str).str.lower() == train_source]
    va_df = df[sp.isin(["val", "valid", "validation", "dev"])]
    te_df = df[sp.isin(["test"])]
    tr = DataLoader(ManifestDS(root, tr_df, train=True, noise_dir=noise, music_dir=music,
                               aug_voice=aug_voice), batch_size=batch,
                    shuffle=True, num_workers=workers, persistent_workers=workers > 0)
    va = DataLoader(ManifestDS(root, va_df, train=False), batch_size=batch,
                    num_workers=workers, persistent_workers=workers > 0)
    return tr, va, te_df, (len(tr_df), len(va_df), len(te_df))


# ============================================================================ #
#  Class GOLD master dataset (HuggingFace / /data/ai231) with OUT_OF_SCOPE      #
# ============================================================================ #
_CLIP_CACHE = {}


def _clip_bank(dir_path, max_per_file=2, max_clips=600):
    """Load a dir's audio ONCE into a single (N, 1, CLIP_LEN) tensor.

    One contiguous tensor (not a list of thousands) so DataLoader workers inherit
    it via copy-on-write cleanly -- a big Python list triggers refcount COW storms
    on fork and freezes worker startup. Capped so the bank stays small & fork-safe.
    """
    if not dir_path:
        return []
    key = os.path.abspath(dir_path)
    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]
    files = []
    for ext in ("*.wav", "*.flac", "*.ogg", "*.mp3"):
        files += glob.glob(os.path.join(dir_path, ext))
    random.shuffle(files)
    clips = []
    for f in files:
        if len(clips) >= max_clips:
            break
        try:
            wav, sr = torchaudio.load(f)
        except Exception:
            continue
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(0, keepdim=True)
        T = wav.shape[-1]
        if T <= CLIP_LEN:
            clips.append(pad_or_trim(wav).contiguous())
        else:
            for _ in range(max_per_file):
                if len(clips) >= max_clips:
                    break
                o = random.randint(0, T - CLIP_LEN)
                clips.append(wav[..., o:o + CLIP_LEN].clone())
    bank = torch.stack(clips) if clips else []     # (N,1,CLIP_LEN) single COW-safe tensor
    _CLIP_CACHE[key] = bank
    print(f"[aug] cached {len(bank)} clips from {dir_path}")
    return bank


class HFSLUDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace split of the class master dataset -> (logmel, intent, slot).

    Maps the dataset's `command` field to our intent (out-of-scope -> OUT_OF_SCOPE).
    Reuses the same noise/music/voice augmentation as ManifestDS.
    """
    _AUDIO = ("audio",)
    _INT = ("command", "intent", "label")
    _SLOT = ("slot_value", "slot")
    _OOS = ("out_of_scope", "oos", "is_out_of_scope")

    def __init__(self, split, train=False, noise_dir=None, music_dir=None, aug_voice=False):
        self.ds = split
        self.train = train
        self.aug_voice = aug_voice
        self.noises = _clip_bank(noise_dir) if (train and noise_dir) else []
        self.music = _clip_bank(music_dir) if (train and music_dir) else []
        self.tmask = torchaudio.transforms.TimeMasking(time_mask_param=25)
        self.fmask = torchaudio.transforms.FrequencyMasking(freq_mask_param=7)
        cols = split.column_names
        self.c_audio = next((k for k in self._AUDIO if k in cols), None)
        self.c_int = next((k for k in self._INT if k in cols), None)
        self.c_slot = next((k for k in self._SLOT if k in cols), None)
        self.c_oos = next((k for k in self._OOS if k in cols), None)
        if not self.c_audio or not self.c_int:
            raise SystemExit(f"HF split missing audio/intent columns; has {cols}")

    def __len__(self):
        return len(self.ds)

    def _load_file(self, path):
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return wav.mean(0, keepdim=True)

    def _decode_audio(self, a):
        """Decode an HF audio cell -> (mono wav (1,T), sr) without torchcodec.
        Handles a decoded {'array',...}, an undecoded {'bytes'/'path'}, or a raw path."""
        if isinstance(a, dict):
            if a.get("array") is not None:
                w = torch.as_tensor(a["array"], dtype=torch.float32)
                w = w.unsqueeze(0) if w.ndim == 1 else w.mean(0, keepdim=True)
                return w, a.get("sampling_rate", SR)
            if a.get("bytes") is not None:
                import io
                import soundfile as sf
                data, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
                return torch.as_tensor(data.T, dtype=torch.float32).mean(0, keepdim=True), sr
            if a.get("path"):
                return self._load_file(a["path"]), SR
        if isinstance(a, str):
            return self._load_file(a), SR
        raise ValueError(f"unrecognized audio field: {type(a)!r}")

    def _random_clip(self, pool):
        return random.choice(pool)          # pool = pre-cached (1, CLIP_LEN) clips

    def _mix(self, wav, bg, lo, hi):
        snr = random.uniform(lo, hi)
        return wav + bg * (wav.norm() / (bg.norm() + 1e-6)) * (10 ** (-snr / 20))

    def __getitem__(self, i):
        row = self.ds[i]
        wav, sr = self._decode_audio(row[self.c_audio])
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if self.train and self.aug_voice and random.random() < 0.5:
            # speed+pitch perturbation via a GCD-friendly resample (10:9 / 10:11 ->
            # tiny kernel = ~ms). Replaces torchaudio.pitch_shift, which is ~1000x
            # slower here (phase vocoder + near-coprime resample kernels).
            wav = torchaudio.functional.resample(
                wav, SR, random.choice([SR * 9 // 10, SR * 11 // 10]))   # 14400 / 17600
        wav = pad_or_trim(wav)
        if self.train:
            wav = torch.roll(wav, random.randint(-1600, 1600), dims=-1)
            if len(self.noises) and random.random() < 0.6:
                wav = self._mix(wav, self._random_clip(self.noises), 5.0, 20.0)
            if len(self.music) and random.random() < 0.3:
                wav = self._mix(wav, self._random_clip(self.music), 8.0, 25.0)
        feat = logmel(wav)
        if self.train:
            feat = self.fmask(self.tmask(feat))
        oos = bool(row[self.c_oos]) if self.c_oos else False
        cmd = str(row[self.c_int]).strip()
        intent = INTENT2IDX[OOS] if (oos or cmd not in INTENT2IDX) else INTENT2IDX[cmd]
        slot = slot_index(row[self.c_slot]) if self.c_slot else slot_index(None)
        return feat, intent, slot


# The class master dataset ships as a HF *cache* copy, not a save_to_disk dump:
#   <root>/<ns>___<name>/<config>/<version>/<hash>/*.arrow
# with three configs -- default (gold: train/test/holdout/numerals),
# supplemental_synth (train), synthetic_negatives (train/test = OUT_OF_SCOPE).
# load_dataset(<root>) tries to MERGE all configs and dies with a CastError
# (supplemental_synth stores audio as struct<bytes,path>, default as Audio).
# So we read each config's arrow files directly and cast audio ourselves.
MASTER_MAIN_CONFIG = "default"
MASTER_EXTRA_CONFIGS = ("supplemental_synth", "synthetic_negatives")


def _cache_config_dir(root, config):
    """Return the <version>/<hash> dir holding a config's arrow files, or None."""
    hits = [d for d in glob.glob(os.path.join(root, "*", config, "*", "*"))
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*.arrow"))]
    return sorted(hits)[-1] if hits else None


def _read_arrow_split(cfg_dir, split):
    """Build one split Dataset from its (possibly sharded) arrow files; cast audio."""
    from datasets import Dataset, concatenate_datasets, Audio
    files = sorted(glob.glob(os.path.join(cfg_dir, f"*-{split}.arrow"))) \
        + sorted(glob.glob(os.path.join(cfg_dir, f"*-{split}-*-of-*.arrow")))
    if not files:
        return None
    parts = [Dataset.from_file(f) for f in files]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if "audio" in ds.column_names:
        # decode=False -> a row returns {'bytes','path'} unchanged; we decode with
        # soundfile in HFSLUDataset, sidestepping datasets>=4's torchcodec requirement.
        ds = ds.cast_column("audio", Audio(sampling_rate=SR, decode=False))
    return ds


def load_master(source, config=MASTER_MAIN_CONFIG, offline=True):
    """Load one config of the class master dataset as {split: Dataset}.

    Handles the HF cache layout at /data/ai231 (read arrow files directly,
    per-config, so mismatched audio encodings never collide). Falls back to
    load_from_disk / load_dataset for a real saved dir or a hub name.
    """
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if os.path.isdir(source):
        cfg_dir = _cache_config_dir(source, config)
        if cfg_dir:
            splits = {s: d for s in ("train", "test", "holdout", "numerals",
                                     "validation", "val")
                      if (d := _read_arrow_split(cfg_dir, s)) is not None}
            if splits:
                return splits
        from datasets import load_dataset, load_from_disk
        try:
            return load_from_disk(source)
        except Exception:
            return load_dataset(source)
    from datasets import load_dataset
    return load_dataset(source)


def build_loaders_hf(source, batch=128, workers=4, noise=None, music=None,
                     aug_voice=False, supplemental=None, val_split="test",
                     extra_configs=MASTER_EXTRA_CONFIGS, neg_oversample=1,
                     real_root=None, real_oversample=1):
    """Loaders from the master dataset. Train = default/train + the bundled
    extra configs (supplemental_synth, synthetic_negatives=OUT_OF_SCOPE) +
    an optional external supplemental dir, combined at the PyTorch level so
    mismatched HF schemas never need reconciling. Validate on `val_split`.
    `neg_oversample` repeats the synthetic_negatives train set N times to
    counter the OUT_OF_SCOPE class imbalance (reject-class recall)."""
    from torch.utils.data import ConcatDataset

    def mk(split, train):
        return HFSLUDataset(split, train=train,
                            noise_dir=noise if train else None,
                            music_dir=music if train else None,
                            aug_voice=aug_voice and train)

    main = load_master(source, MASTER_MAIN_CONFIG)
    train_sets = [mk(main["train"], True)]

    if os.path.isdir(source):                       # bundled extra configs
        for cfg in extra_configs:
            try:
                d = load_master(source, cfg)
                if "train" in d:
                    reps = neg_oversample if cfg == "synthetic_negatives" else 1
                    for _ in range(max(1, reps)):
                        train_sets.append(mk(d["train"], True))
                    print(f"added config '{cfg}': +{len(d['train'])} train clips"
                          f"{f' x{reps}' if reps > 1 else ''}")
            except Exception as e:
                print(f"[warn] config '{cfg}' skipped ({repr(e)[:120]})")

    if supplemental:                                # optional external dir
        try:
            sup = load_master(supplemental)
            sp = sup["train"] if "train" in sup else sup[list(sup.keys())[0]]
            train_sets.append(mk(sp, True))
            print(f"added supplemental: +{len(sp)} train clips")
        except Exception as e:
            print(f"[warn] supplemental not merged ({repr(e)[:120]}); continuing")

    if real_root:                                   # YOUR real mic recordings (fine-tune)
        import pandas as pd
        rdf = pd.read_csv(os.path.join(real_root, "manifest.csv"))
        for _ in range(max(1, real_oversample)):
            train_sets.append(ManifestDS(real_root, rdf, train=True, noise_dir=noise,
                                         music_dir=music, aug_voice=aug_voice))
        print(f"added {len(rdf)} REAL recordings x{real_oversample}")

    train_ds = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val = main[val_split] if val_split in main else main["test"]
    val_ds = mk(val, False)
    tr = DataLoader(train_ds, batch_size=batch, shuffle=True,
                    num_workers=workers, persistent_workers=workers > 0)
    va = DataLoader(val_ds, batch_size=batch,
                    num_workers=workers, persistent_workers=workers > 0)
    return tr, va, (len(train_ds), len(val_ds))
