"""Log-Mel front-end shared by training, inference and benchmarking."""
import numpy as np
import torch
import torchaudio

from config import SR, N_MELS, N_FFT, WIN, HOP, CLIP_LEN

# One shared transform instance (stateless, safe to reuse).
_melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
    n_mels=N_MELS, power=2.0,
)


def pad_or_trim(wav: torch.Tensor) -> torch.Tensor:
    """Force a (channels, T) waveform to exactly CLIP_LEN samples."""
    if wav.shape[-1] < CLIP_LEN:
        wav = torch.nn.functional.pad(wav, (0, CLIP_LEN - wav.shape[-1]))
    return wav[..., :CLIP_LEN]


def logmel(wav: torch.Tensor) -> torch.Tensor:
    """(1, T) waveform in [-1, 1]  ->  (1, N_MELS, N_FRAMES) log-mel features.

    Peak-normalizes first so the features are invariant to recording gain -- the
    training clips sit near full-scale but a live mic is much quieter, and without
    this the model sees out-of-distribution (too-quiet) input and predicts a flat
    softmax. Peak (not RMS) because zero-padding would skew an RMS estimate.
    """
    wav = pad_or_trim(wav)
    peak = wav.abs().max()
    if peak > 1e-3:
        wav = wav / peak
    mel = _melspec(wav)                 # (1, N_MELS, N_FRAMES)
    return torch.log(mel + 1e-6)


def wav_to_input(wav_np: np.ndarray) -> np.ndarray:
    """float32 mono numpy waveform -> (1, 1, N_MELS, N_FRAMES) float32 for ONNX."""
    wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32)).unsqueeze(0)
    feat = logmel(wav)                  # (1, N_MELS, N_FRAMES)
    return feat.unsqueeze(0).numpy()    # (1, 1, N_MELS, N_FRAMES)
