"""
eeg_decoder.py
================================================================================
A compact EEGNet-style convolutional network (PyTorch), trained to decode task
condition or group directly from raw multichannel EEG epochs, with a rigorous
subject-aware evaluation harness.

What this demonstrates:
  - A custom nn.Module implementing the EEGNet design from scratch: a temporal
    convolution, a depthwise spatial convolution, and a separable convolution,
    with batchnorm / ELU / dropout and an optional max-norm constraint.
  - A correct training loop: Adam, class-weighted loss for imbalance, LR
    scheduling, and early stopping on a held-out validation split.
  - Honest evaluation: GroupKFold by participant, so the network is always tested
    on people it never trained on (no subject leakage). Reports balanced accuracy
    and a confusion matrix.

Reference: Lawhern et al. (2018), "EEGNet: a compact convolutional network for
EEG-based brain-computer interfaces." This is an independent implementation.

Run on demo data (no real EEG needed):
  python make_demo_data.py
  python eeg_decoder.py --real demo_data --target condition

Targets:
  condition  -> ego / allo / control   (3-way)
  ego        -> egocentric vs the rest  (binary; the most behaviorally distinct)
  group      -> deaf vs hearing         (binary)

Dependencies: torch, numpy, scipy, scikit-learn
================================================================================
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from eeg_data import load_real_epochs_full, GROUP_NAMES

torch.set_num_threads(7)


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

class Conv2dMaxNorm(nn.Conv2d):
    """Conv2d with an optional max-norm constraint on its weights (EEGNet uses this)."""
    def __init__(self, *a, max_norm=None, **k):
        super().__init__(*a, **k)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                norm = torch.linalg.vector_norm(self.weight, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
                desired = norm.clamp(max=self.max_norm)
                self.weight.mul_(desired / norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """
    Input:  (N, C, T)   ->  treated as a single-channel (1, C, T) image.
    Output: (N, n_classes) logits.
    """
    def __init__(self, n_channels, n_times, n_classes,
                 F1=8, D=2, F2=16, kern_length=64, dropout=0.5):
        super().__init__()
        # Block 1: temporal conv -> depthwise spatial conv (over all channels)
        self.conv_temporal = nn.Conv2d(1, F1, (1, kern_length),
                                       padding=(0, kern_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv_depthwise = Conv2dMaxNorm(F1, F1 * D, (n_channels, 1),
                                            groups=F1, bias=False, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv (depthwise temporal + pointwise mix)
        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16),
                                        padding=(0, 8), groups=F1 * D, bias=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier
        feat_t = n_times // 32   # two pools: /4 then /8
        self.fc = Conv2dMaxNorm(F2, n_classes, (1, max(1, feat_t)),
                                bias=True, max_norm=0.25)

    def forward(self, x):
        x = x.unsqueeze(1)                       # (N,1,C,T)
        x = self.bn1(self.conv_temporal(x))
        x = F.elu(self.bn2(self.conv_depthwise(x)))
        x = self.drop1(self.pool1(x))
        x = self.conv_sep_point(self.conv_sep_depth(x))
        x = F.elu(self.bn3(x))
        x = self.drop2(self.pool2(x))            # (N,F2,1,feat_t)
        x = self.fc(x)                           # (N,n_classes,1,1)
        return x.flatten(1)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def class_weights(y, n_classes, device):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_fold(Xtr, ytr, subj_tr, Xte, yte, n_classes, cfg, device):
    # carve a validation set out of the training subjects (no leakage)
    uniq = np.unique(subj_tr)
    rng = np.random.default_rng(0)
    val_subj = set(rng.choice(uniq, max(1, len(uniq) // 5), replace=False))
    vmask = np.array([s in val_subj for s in subj_tr])

    Xt = torch.tensor(Xtr[~vmask], dtype=torch.float32)
    yt = torch.tensor(ytr[~vmask], dtype=torch.long)
    Xv = torch.tensor(Xtr[vmask], dtype=torch.float32, device=device)
    yv = torch.tensor(ytr[vmask], dtype=torch.long, device=device)

    model = EEGNet(Xtr.shape[1], Xtr.shape[2], n_classes,
                   dropout=cfg.dropout, kern_length=cfg.kern_length).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr[~vmask], n_classes, device))

    n = Xt.shape[0]
    best_val, best_state, wait = -1.0, None, 0
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            vpred = model(Xv).argmax(1).cpu().numpy()
        vbacc = balanced_accuracy_score(yv.cpu().numpy(), vpred)
        sched.step(-vbacc)
        if vbacc > best_val:
            best_val, best_state, wait = vbacc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=str, required=True)
    ap.add_argument("--target", choices=["condition", "ego", "group"], default="condition")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--kern_length", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    cfg = ap.parse_args()
    if cfg.smoke:
        cfg.epochs, cfg.folds, cfg.patience = 4, 2, 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    d = load_real_epochs_full(cfg.real)
    x, cond, group, subject = d["x"], d["cond"], d["group"], d["subject"]

    if cfg.target == "condition":
        y, names = cond, ["ego", "allo", "control"]
    elif cfg.target == "ego":
        y, names = (cond == 0).astype(int), ["rest", "ego"]
    else:
        y, names = (group >= 2).astype(int), ["deaf", "hearing"]
    n_classes = len(names)
    chance = 1.0 / n_classes
    print(f"Target: {cfg.target}  classes={names}  counts={np.bincount(y, minlength=n_classes)}")

    gkf = GroupKFold(n_splits=cfg.folds)
    all_true, all_pred, baccs = [], [], []
    for k, (tr, te) in enumerate(gkf.split(x, y, groups=subject)):
        pred = train_fold(x[tr], y[tr], subject[tr], x[te], y[te], n_classes, cfg, device)
        b = balanced_accuracy_score(y[te], pred)
        baccs.append(b)
        all_true.append(y[te]); all_pred.append(pred)
        print(f"  fold {k+1}/{cfg.folds}: balanced accuracy {b:.3f}")

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    print("\n" + "=" * 56)
    print(f"EEGNet decoding: {cfg.target}   (subjects held out)")
    print("=" * 56)
    print(f"  balanced accuracy: {np.mean(baccs):.3f} +/- {np.std(baccs):.3f}  (chance {chance:.3f})")
    print("\n  confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(all_true, all_pred)
    print("           " + "".join(f"{n:>9s}" for n in names))
    for i, row in enumerate(cm):
        print(f"  {names[i]:8s} " + "".join(f"{v:9d}" for v in row))


if __name__ == "__main__":
    main()
