"""
make_demo_data.py  (eeg-diffusion)
================================================================================
Writes synthetic per-participant .mat files in the same format export_epochs.m
produces, so the diffusion pipeline runs end-to-end without real EEG data.

The synthetic data has real structure baked in: a group-dependent mu-power shift
and a condition-dependent mu suppression, so the model has something genuine to
learn and the evaluation has a pattern to recover.

  python make_demo_data.py            # writes ./demo_data/*.mat
  python eeg_decoder.py --real demo_data --target condition
================================================================================
"""

import os
import numpy as np
from scipy.io import savemat

OUT = "demo_data"
# 10-channel sensorimotor-style montage (names the loader's default ROI expects)
CHANS = ["C1", "C2", "C3", "C4", "C5", "C6", "Cz", "CP1", "CP2", "CPz"]
GROUPS = {"DF": 1.20, "DNF": 0.95, "HF": 0.80, "HNF": 1.25, "HNS": 1.00}  # mu gain by group
N_PER_GROUP = {"DF": 6, "DNF": 4, "HF": 5, "HNF": 5, "HNS": 6}
MU_BY_COND = {0: 1.0, 1: 0.6, 2: 1.25}   # ego / allo / control mu suppression


def make_subject(pid, group_gain, rng, fs=500.0, n_trials=130, T=1000):
    t = np.arange(T) / fs
    nav = rng.integers(0, 3, n_trials)
    data = np.zeros((n_trials, len(CHANS), T), dtype=np.float32)
    for i in range(n_trials):
        c = int(nav[i])
        for ch in range(len(CHANS)):
            noise = rng.standard_normal(T)
            freqs = np.fft.rfftfreq(T, 1 / fs)
            background = np.fft.irfft(np.fft.rfft(noise) / np.sqrt(np.maximum(freqs, 1.0)), n=T)
            mu_amp = group_gain * MU_BY_COND[c] * (1.0 if ch < len(CHANS) // 2 else 0.4)
            mu = mu_amp * np.sin(2 * np.pi * 11 * t + rng.uniform(0, 2 * np.pi))
            theta = 0.6 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
            data[i, ch] = (0.6 * background + 0.5 * theta + 0.6 * mu).astype(np.float32)
    rt = 1.0 + 0.3 * (nav == 0) + rng.normal(0, 0.1, n_trials)   # ego slower
    acc = (rng.random(n_trials) < np.where(nav == 0, 0.65, 0.95)).astype(float)
    return dict(data=data, nav=nav.astype(float).reshape(-1, 1),
                dff=nav.astype(float).reshape(-1, 1),
                acc=acc.reshape(-1, 1), rt=rt.reshape(-1, 1), srate=fs,
                chanlabels=np.array(CHANS, dtype=object).reshape(1, -1),
                group="demo", pid=pid)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 0
    for prefix, gain in GROUPS.items():
        for k in range(1, N_PER_GROUP[prefix] + 1):
            pid = f"{prefix}{k:02d}"
            savemat(os.path.join(OUT, f"{pid}.mat"), make_subject(pid, gain, rng))
            n += 1
    print(f"wrote {n} synthetic participants to ./{OUT}/")
    print("now run:  python eeg_decoder.py --real demo_data --target condition")


if __name__ == "__main__":
    main()
