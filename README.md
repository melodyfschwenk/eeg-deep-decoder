# Deep Learning EEG Decoder (PyTorch, EEGNet)

A compact EEGNet-style convolutional network implemented from scratch in
PyTorch, trained to decode task condition or group **directly from raw
multichannel EEG epochs**, with a subject-aware evaluation harness.

The point of the project is to show end-to-end PyTorch competence on a real,
messy modality: a custom convolutional architecture, a correct training loop
with class weighting / LR scheduling / early stopping, and an evaluation that
holds whole participants out so reported accuracy reflects real generalization.

## Architecture

An independent implementation of EEGNet (Lawhern et al., 2018):

1. **Temporal convolution** learns frequency filters across time.
2. **Depthwise spatial convolution** (one kernel spanning all channels) learns
   spatial patterns per temporal filter, with a max-norm weight constraint.
3. **Separable convolution** (depthwise temporal + pointwise mixing) summarizes
   each feature map compactly.
4. BatchNorm, ELU, average pooling, and dropout throughout; a max-norm
   constrained classifier head.

It runs on raw epochs `(trials, channels, time)`, no hand-crafted features.

## Training harness

- **GroupKFold by participant:** the network is always tested on people it never
  trained on. A validation split is carved from the *training* participants for
  early stopping, so the test fold is never touched during model selection.
- **Class-weighted loss** handles condition imbalance (control is the minority).
- **Adam + ReduceLROnPlateau + early stopping** on validation balanced accuracy.
- **GPU autodetected**; runs CPU-only without changes.
- Reports per-fold and mean balanced accuracy plus a confusion matrix.

## Run it (no real data needed)

```bash
pip install -r requirements.txt
python make_demo_data.py
python eeg_decoder.py --real demo_data --target condition
```

On the synthetic demo (which has a real condition effect baked in) the decoder
lands well above chance, confirming the pipeline learns genuine structure. A
fast wiring check is `--smoke`.

## Targets

```bash
python eeg_decoder.py --real demo_data --target condition   # ego/allo/control (3-way)
python eeg_decoder.py --real demo_data --target ego         # egocentric vs rest (binary)
python eeg_decoder.py --real demo_data --target group       # deaf vs hearing (binary)
```

`ego` is often the most decodable target because egocentric trials are the most
behaviorally distinct.

## Run it on your own exported epochs

```bash
python eeg_decoder.py --real /path/to/export --target condition --epochs 100
```

`export_epochs.m` produces the per-participant `.mat` epochs the loader expects.

## Files

| file | role |
|------|------|
| `eeg_decoder.py` | EEGNet implementation + subject-aware training/eval harness |
| `eeg_data.py` | loads/pools exported `.mat` epochs |
| `export_epochs.m` | MATLAB: EEGLAB `.set` -> labeled `.mat` epochs |
| `make_demo_data.py` | writes synthetic demo data so the repo runs without real EEG |

## Notes on honesty

Single-trial EEG decoding across unseen participants is genuinely hard, and
balanced accuracy on real data is often modest and above chance rather than
high. That is expected for this modality and sample size; the value here is a
correct, well-regularized, leakage-free pipeline. Real human-subjects data is
not included; the repo ships only synthetic demo data.
