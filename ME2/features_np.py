"""Torch-free log-mel front-end (NumPy only) -- drop-in replacement for features.py.

PyPI's torch wheels use ARM instructions the Raspberry Pi 4 (Cortex-A72) lacks and
crash with "Illegal instruction". This reproduces exactly what the model trained on:
torchaudio.transforms.MelSpectrogram(n_fft=400, hop=160, win=400, n_mels=40,
power=2.0, hann-periodic window, center + reflect pad, HTK mel, norm=None) followed
by peak-normalization and log(mel + 1e-6).

Same public API as features.py (pad_or_trim / logmel / wav_to_input), so copying
this over the Pi's features.py makes the whole inference path torch-free.
Validate it matches the torch version before deploying (see README note below):
    python -c "import numpy as np,features,features_np as f; w=np.random.randn(30000).astype('float32')*0.1; print('maxdiff', abs(features.wav_to_input(w)-f.wav_to_input(w)).max())"
"""
import numpy as np

from config import SR, N_MELS, N_FFT, HOP, CLIP_LEN

_FMAX = SR / 2.0


def _hann_periodic(n):
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def _mel_fb():
    """HTK mel filterbank, norm=None -- matches torchaudio.functional.melscale_fbanks."""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs = N_FFT // 2 + 1
    all_freqs = np.linspace(0.0, SR // 2, n_freqs)
    m_pts = np.linspace(hz2mel(0.0), hz2mel(_FMAX), N_MELS + 2)
    f_pts = mel2hz(m_pts)
    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - all_freqs[:, None]            # (n_freqs, n_mels+2)
    down = -slopes[:, :-2] / f_diff[:-1][None, :]
    up = slopes[:, 2:] / f_diff[1:][None, :]
    fb = np.maximum(0.0, np.minimum(down, up))              # (n_freqs, n_mels)
    return fb.astype(np.float32)


_WINDOW = _hann_periodic(N_FFT)
_FB = _mel_fb()


def pad_or_trim(wav):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.shape[0] < CLIP_LEN:
        wav = np.pad(wav, (0, CLIP_LEN - wav.shape[0]))
    return wav[:CLIP_LEN]


def logmel(wav):
    """1-D waveform -> (1, N_MELS, N_FRAMES) log-mel, matching the torch front-end."""
    wav = pad_or_trim(wav)
    peak = np.abs(wav).max()
    if peak > 1e-3:
        wav = wav / peak
    pad = N_FFT // 2                                        # center=True
    x = np.pad(wav, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * _WINDOW[None, :]                      # (frames, n_fft)
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)            # (frames, n_freqs)
    power = (spec.real ** 2 + spec.imag ** 2)              # |.|^2
    mel = power @ _FB                                       # (frames, n_mels)
    logm = np.log(mel.T + 1e-6)                             # (n_mels, frames)
    return logm[None, :, :].astype(np.float32)             # (1, N_MELS, N_FRAMES)


def wav_to_input(wav_np):
    """float32 mono numpy waveform -> (1, 1, N_MELS, N_FRAMES) float32 for ONNX."""
    return logmel(wav_np)[None, :, :, :]
