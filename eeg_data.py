"""
load_eeg.py
================================================================================
Loads per-participant .mat files from export_epochs.m, selects an ROI, resamples,
pools everyone. Two entry points:

  load_real_epochs(dir, ...)        -> x, cond, group   (or + subject)
  load_real_epochs_full(dir, ...)   -> dict with x, cond, group, subject, acc, rt

Groups (from filename):
  0 Deaf Fluent  1 Deaf Non-Fluent  2 Hearing Fluent  3 Hearing Non-Fluent  4 Hearing Non-Signer
Conditions: 0 ego, 1 allo, 2 control.
================================================================================
"""

import glob
import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample

SENSORIMOTOR = ["C1", "C2", "C3", "C4", "C5", "C6", "Cz", "CP1", "CP2", "CPz"]
FRONTAL = ["Fp1","Fp2","F1","F2","F3","F4","F5","F6","F7","F8","FC1","FC2","FC3","FC4","FC5","FC6","Fz","FCz"]
PARIETAL = ["P1","P2","P3","P4","P5","P6","P7","P8","CP1","CP2","CP3","CP4","CP5","CP6","Pz","CPz"]
GROUP_NAMES = ["Deaf Fluent", "Deaf Non-Fluent",
               "Hearing Fluent", "Hearing Non-Fluent", "Hearing Non-Signer"]
_PREFIX = [("DNF", 1), ("HNF", 3), ("HNS", 4), ("DF", 0), ("HF", 2)]


def _group_code(pid):
    pid = str(pid).upper()
    for prefix, code in _PREFIX:
        if pid.startswith(prefix):
            return code
    return -1


def _chanlabels(m):
    return [str(np.asarray(c).ravel()[0]) for c in np.asarray(m["chanlabels"]).ravel()]


def _core(export_dir, target_sfreq, roi, verbose):
    files = sorted(glob.glob(os.path.join(export_dir, "*.mat")))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {export_dir}")

    # Pass 1: channels present in EVERY participant (some had bad channels removed).
    common = None
    for fp in files:
        labels = set(_chanlabels(loadmat(fp, variable_names=["chanlabels"])))
        common = labels if common is None else (common & labels)
    # final selection: roi channels that survive in everyone, in roi order
    sel = [ch for ch in roi if ch in common]
    dropped = [ch for ch in roi if (ch not in common) and any(ch in _chanlabels(loadmat(f, variable_names=["chanlabels"])) for f in files)]
    if verbose and dropped:
        print(f"  note: dropped {len(dropped)} ROI channel(s) not present in all participants: {dropped}")
    if not sel:
        raise SystemExit("No ROI channels are common to all participants.")

    X, COND, GROUP, SUBJ, ACC, RT, PIDS = [], [], [], [], [], [], []
    subj_id = 0
    sel_labels = sel
    for fp in files:
        m = loadmat(fp)
        data = np.asarray(m["data"], dtype=np.float32)
        nav = np.asarray(m["nav"]).ravel().astype(int)
        srate = float(np.asarray(m["srate"]).ravel()[0])
        labels = _chanlabels(m)
        pid = str(np.asarray(m["pid"]).ravel()[0])
        g = _group_code(pid)
        if g < 0:
            if verbose:
                print(f"  skip {os.path.basename(fp)}: unrecognized group")
            continue
        acc = np.asarray(m["acc"]).ravel().astype(float) if "acc" in m else np.full(len(nav), np.nan)
        rt = np.asarray(m["rt"]).ravel().astype(float) if "rt" in m else np.full(len(nav), np.nan)

        idx = [labels.index(ch) for ch in sel]   # same channels, same order, for everyone
        data = data[:, idx, :]
        if target_sfreq and target_sfreq != srate:
            n_new = int(round(data.shape[2] * target_sfreq / srate))
            data = resample(data, n_new, axis=2).astype(np.float32)
        if data.shape[2] % 2 == 1:
            data = data[:, :, :-1]

        keep = np.isin(nav, [0, 1, 2])
        data, nav, acc, rt = data[keep], nav[keep], acc[keep], rt[keep]
        if data.shape[0] == 0:
            continue

        X.append(data); COND.append(nav); GROUP.append(np.full(len(nav), g))
        SUBJ.append(np.full(len(nav), subj_id)); ACC.append(acc); RT.append(rt)
        PIDS.append(pid)
        subj_id += 1
        if verbose:
            print(f"  {os.path.basename(fp):10s} {data.shape[0]:4d} trials  "
                  f"{data.shape[1]} ch  {data.shape[2]} samples  ({GROUP_NAMES[g]})")

    x = np.concatenate(X, 0)
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True) + 1e-6
    x = ((x - mean) / std).astype(np.float32)
    out = dict(
        x=x,
        cond=np.concatenate(COND, 0).astype("int64"),
        group=np.concatenate(GROUP, 0).astype("int64"),
        subject=np.concatenate(SUBJ, 0).astype("int64"),
        acc=np.concatenate(ACC, 0),
        rt=np.concatenate(RT, 0),
        chans=sel_labels,
        pids=PIDS,
    )
    if verbose:
        print(f"TOTAL {x.shape}  conditions(ego/allo/ctrl)={np.bincount(out['cond'], minlength=3)}")
        print(f"  groups={dict(zip(GROUP_NAMES, np.bincount(out['group'], minlength=5)))}")
        print(f"  behavior present: rt {np.isfinite(out['rt']).sum()} / {len(out['rt'])} trials")
    return out


def load_real_epochs(export_dir, target_sfreq=128, roi=SENSORIMOTOR, verbose=True, return_subject=False):
    d = _core(export_dir, target_sfreq, roi, verbose)
    if return_subject:
        return d["x"], d["cond"], d["group"], d["subject"]
    return d["x"], d["cond"], d["group"]


def load_real_epochs_full(export_dir, target_sfreq=128, roi=SENSORIMOTOR, verbose=True):
    return _core(export_dir, target_sfreq, roi, verbose)
