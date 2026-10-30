"""Příznaky hlasu (samohláska /a/) počítané vlastními algoritmy.

Skupiny příznaků:
  * časová oblast:  F0 (autokorelace), jitter, shimmer, HNR, ZCR
  * frekvenční:     těžiště, rozptyl, roll-off, plochost spektra, poměry energií v pásmech
  * kepstrální:     CPP (cepstral peak prominence), MFCC (průměr a směrodatná odchylka)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from dsp import (autocorrelation, frame_signal, interpolate_nans, lowpass,
                 mean_power_spectrum, mfcc, real_cepstrum, remove_dc)

F0_MIN, F0_MAX = 70.0, 400.0
VOICED_THRESHOLD = 0.3
OCTAVE_RATIO = 0.9    # kandidát na periodu musí mít alespoň tento podíl maxima autokorelace (1.0 = prosté argmax)
N_MFCC = 12


def stable_segment(x: np.ndarray, fs: float) -> np.ndarray:
    """Aktivní část nahrávky bez náběhu a doznění (energetická detekce + ořez okrajů)."""
    size = int(0.02 * fs)
    rms = np.sqrt(np.mean(frame_signal(x, size, size) ** 2, axis=1))
    active = np.flatnonzero(rms > 0.1 * rms.max())
    seg = x[active[0] * size:(active[-1] + 1) * size]
    trim = int(0.2 * fs)
    return seg[trim:-trim] if len(seg) > 4 * trim else seg


def pitch_and_hnr(x: np.ndarray, fs: float, frame_ms: float = 40.0, hop_ms: float = 20.0):
    """F0 a HNR po rámcích z normované autokorelace. -> (f0[Hz], hnr[dB], podíl znělých rámců)."""
    size, hop = int(fs * frame_ms / 1000), int(fs * hop_ms / 1000)
    lag_lo, lag_hi = int(fs / F0_MAX), int(fs / F0_MIN)
    f0, hnr = [], []
    frames = frame_signal(x, size, hop)
    for frame in frames:
        frame = frame - frame.mean()
        if np.allclose(frame, 0):
            continue
        r = autocorrelation(frame * np.hanning(size))
        r = r / np.maximum(autocorrelation(np.hanning(size)), 1e-6)  # korekce vlivu okna
        seg = r[lag_lo:lag_hi + 1]
        if seg.max() < VOICED_THRESHOLD:
            continue
        # Prosté argmax volí často násobek periody (r(2T) ≈ r(T), korekce oknem dlouhá zpoždění ještě
        # zesiluje) -> F0 o oktávu níž. Bere se proto první lokální maximum s výškou >= 90 % maxima.
        peaks, _ = find_peaks(seg)
        strong = peaks[seg[peaks] >= OCTAVE_RATIO * seg.max()] if peaks.size else []
        k = int(strong[0] if len(strong) else np.argmax(seg)) + lag_lo
        if k <= 0 or k >= size - 1:
            continue
        # parabolická interpolace píku
        a, b, c = r[k - 1], r[k], r[k + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        f0.append(fs / (k + shift))
        peak = np.clip(b, 1e-3, 0.999)
        hnr.append(10 * np.log10(peak / (1 - peak)))
    return np.array(f0), np.array(hnr), len(f0) / max(len(frames), 1)


def jitter_shimmer(x: np.ndarray, fs: float, f0: float) -> dict[str, float]:
    """Jitter (local, RAP) a shimmer (local, APQ3) z jednotlivých period hlasivek."""
    nan = {"jitter_local": np.nan, "jitter_rap": np.nan, "shimmer_local": np.nan, "shimmer_apq3": np.nan}
    if not np.isfinite(f0):
        return nan
    smooth = lowpass(x, fs, 1000.0)
    idx, _ = find_peaks(smooth, distance=max(int(0.7 * fs / f0), 1), height=0.3 * np.max(np.abs(smooth)))
    if len(idx) < 8:
        return nan
    # sub-vzorkové zpřesnění pozic a amplitud (parabola)
    a, b, c = smooth[idx - 1], smooth[idx], smooth[idx + 1]
    denom = a - 2 * b + c
    shift = np.where(denom != 0, 0.5 * (a - c) / np.where(denom == 0, 1, denom), 0.0)
    t = (idx + shift) / fs
    amp = b - 0.25 * (a - c) * shift
    period = np.diff(t)
    good = np.abs(period - np.median(period)) < 0.3 * np.median(period)
    p2, p3 = good[:-1] & good[1:], good[:-2] & good[1:-1] & good[2:]
    if p2.sum() < 5 or p3.sum() < 5:
        return nan
    mean_t, mean_a = period[good].mean(), amp[:-1][good].mean()
    return {
        "jitter_local": np.mean(np.abs(np.diff(period))[p2]) / mean_t,
        "jitter_rap": np.mean(np.abs(period[1:-1] - (period[:-2] + period[1:-1] + period[2:]) / 3)[p3]) / mean_t,
        "shimmer_local": np.mean(np.abs(np.diff(amp))[good]) / mean_a,
        "shimmer_apq3": np.mean(np.abs(amp[1:-1] - (amp[:-2] + amp[1:-1] + amp[2:]) / 3)[p2]) / mean_a,
    }


def spectral_features(x: np.ndarray, fs: float) -> dict[str, float]:
    """Příznaky z průměrného výkonového spektra (FFT)."""
    freq, power = mean_power_spectrum(x, fs)
    total = power.sum() + 1e-12
    centroid = (freq * power).sum() / total
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, fs / 2)]
    energy = [power[(freq >= lo) & (freq < hi)].sum() / total for lo, hi in bands]
    cumulative = np.cumsum(power) / total
    useful = power[1:] + 1e-12
    return {
        "spec_centroid": centroid,
        "spec_spread": np.sqrt((((freq - centroid) ** 2) * power).sum() / total),
        "spec_rolloff85": freq[np.searchsorted(cumulative, 0.85)],
        "spec_flatness": np.exp(np.mean(np.log(useful))) / np.mean(useful),
        **{f"band_{lo:.0f}_{hi:.0f}": e for (lo, hi), e in zip(bands, energy)},
        "low_high_ratio_db": 10 * np.log10((energy[0] + energy[1] + 1e-12) / (energy[2] + energy[3] + 1e-12)),
    }


def cepstral_peak_prominence(frame: np.ndarray, fs: float) -> float:
    """CPP: výška kepstrálního píku nad regresní přímkou (oblast kvefrencí F0)."""
    quef, ceps = real_cepstrum(frame, fs)
    sel = (quef >= 1 / F0_MAX) & (quef <= 1 / F0_MIN)
    q, c = quef[sel], ceps[sel]
    slope, intercept = np.polyfit(q, c, 1)
    k = int(np.argmax(c))
    return float(c[k] - (slope * q[k] + intercept))


def extract_features(x: np.ndarray, fs: float) -> dict[str, float]:
    """Vektor příznaků jedné nahrávky."""
    x = remove_dc(interpolate_nans(x))
    x = stable_segment(x, fs)
    x = x / np.max(np.abs(x))

    f0, hnr, voiced_ratio = pitch_and_hnr(x, fs)
    f0_mean = float(np.median(f0)) if len(f0) else np.nan
    feats: dict[str, float] = {
        "f0_mean": f0_mean,
        "f0_std": float(np.std(f0)) if len(f0) else np.nan,
        "voiced_ratio": voiced_ratio,
        "hnr_mean": float(np.mean(hnr)) if len(hnr) else np.nan,
        "hnr_std": float(np.std(hnr)) if len(hnr) else np.nan,
        "zcr": float(np.mean(np.abs(np.diff(np.sign(x)))) / 2),
    }
    feats.update(jitter_shimmer(x, fs, f0_mean))
    feats.update(spectral_features(x, fs))

    frames = frame_signal(x, 512, 256)
    cpp = [cepstral_peak_prominence(f, fs) for f in frames]
    feats["cpp_mean"], feats["cpp_std"] = float(np.mean(cpp)), float(np.std(cpp))

    coeffs = mfcc(x, fs, n_mfcc=N_MFCC + 1)[:, 1:]  # bez c0 (energie)
    for i in range(N_MFCC):
        feats[f"mfcc{i + 1}_mean"] = float(coeffs[:, i].mean())
        feats[f"mfcc{i + 1}_std"] = float(coeffs[:, i].std())
    return feats
