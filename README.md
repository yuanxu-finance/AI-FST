# AI-FST

Research implementation of **EXACT: Explicit Noise-Structure and Cross-Granularity Disentanglement for Context-Aware Financial Volatility Forecasting**.

EXACT combines a topology-aware encoder (**TopoEnc**), a decomposition-guided separator (**DecompSep**), and context-aware fusion (**CtxFuse**) for financial volatility forecasting. The repository name is AI-FST; the model name remains EXACT.

## Repository structure

```text
AI-FST/
├── datasets/              # Eight datasets, one directory per dataset
├── model/                 # EXACT.py: complete neural pipeline
├── training/              # Preprocessing, losses, training, and evaluation
├── main.py                # Configuration and command-line entry point
├── requirements.txt
└── requirements-tested.txt
```

The organization follows the `datasets/`, `model/`, and `main.py` pattern of [HiCAPM](https://github.com/AI-Recsys/HiCAPM). Its recommendation algorithms and code are not used here. The additional `training/` package keeps the two financial-data protocols separate from the neural architecture.

## Requirements

Use Python 3.10 or 3.11. Install a PyTorch build appropriate for your machine, then:

```bash
pip install -r requirements.txt
```

The release checks use PyTorch 2.1.1 with CUDA 11.8 and the versions in `requirements-tested.txt`. CPU execution is supported; CUDA is recommended for full experiments. Persistent homology uses Ripser. VMD is implemented in the supplied preprocessing code and does not require `vmdpy`.

## Quick start

List datasets:

```bash
python main.py --list_datasets
```

Check every CSV and its temporal split without training:

```bash
python main.py --data all --check_data
```

Train with the retained experiment configuration:

```bash
python main.py --data AAPL
python main.py --data BTC
```

Run a short check, or select a custom dataset root:

```bash
python main.py --data AAPL --epochs 1 --device cpu
python main.py --data Amazon --data_dir /path/to/datasets
```


## Main options

| Option | Purpose |
|---|---|
| `--data` | `AAPL`, `Amazon`, `Google`, `Nasdaq`, `NYSE`, `CSI300`, `SP500`, `BTC`, or `all` |
| `--data_dir` | Dataset root; defaults to this repository's `datasets/` |
| `--epochs`, `--patience` | Override training duration and early stopping |
| `--batch_size` | Override the mini-batch size |
| `--lr`, `--weight_decay` | Override AdamW settings |
| `--seed` | Random seed, default `42` |
| `--device` | `auto`, `cpu`, or `cuda` |
| `--output_dir` | Output root, default `outputs/` |
| `--check_data` | Validate inputs and report window counts without training |

Without overrides, daily experiments use the original dataset-specific hyperparameter profiles; BTC uses its own minute-track defaults. `python main.py --help` lists all options.

## Model components

- **TopoEnc:** delay embedding and persistent homology produce Betti-curve descriptors, which are projected into a learned topological representation.
- **DecompSep:** VMD, permutation-entropy screening, and frequency partitioning construct reconstruction, persistent, high-frequency, and noise targets for auxiliary supervision.
- **CtxFuse:** fine and coarse attention backbones are combined through an element-wise gate conditioned on state, time, topology, and backbone features; topology also supplies FiLM modulation.

The complete neural pipeline is in `model/EXACT.py`. Daily and minute protocols pass their settings explicitly to the same model. Preprocessing and training remain in `training/`.

## Evaluation

The principal metrics are **MSE, RMSE, MAE, R², and MAPE**, computed after inverse standardization on the log realized-variance scale. Each run writes:

```text
outputs/<dataset>/
├── run_config.json
├── summary.csv
├── history.csv
└── predictions.csv       # BTC track
```

The BTC track additionally reports direction accuracy, QLIKE, RV-scale SMAPE, and Pearson correlation. The original daily runner does not export per-timestamp predictions or model checkpoints. Results retain legacy model identifiers from the research runners.

This repository provides the full EXACT pipeline and eight input datasets. Baseline and ablation implementations are not included. Execution checks do not constitute reproduction of all manuscript results.
