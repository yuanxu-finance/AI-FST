import copy
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from model import EXACT

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import ripser
except ImportError as exc:
    raise ImportError("Missing dependency: 'ripser'. Install via 'pip install ripser'.") from exc


OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "outputs")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    train_stride: int
    test_stride: int


DATA_DIR = str(Path(__file__).resolve().parents[1] / "datasets")
DATASETS = [
    DatasetSpec("AAPL", os.path.join(DATA_DIR, "AAPL.csv"), 1, 1),
    DatasetSpec("Amazon", os.path.join(DATA_DIR, "Amazon.csv"), 1, 1),
    DatasetSpec("Google", os.path.join(DATA_DIR, "Google.csv"), 1, 1),
    DatasetSpec("Nasdaq", os.path.join(DATA_DIR, "Nasdaq.csv"), 1, 1),
    DatasetSpec("NYSE", os.path.join(DATA_DIR, "NYSE.csv"), 1, 1),
    DatasetSpec("CSI 300", os.path.join(DATA_DIR, "CSI300.csv"), 1, 1),
    DatasetSpec("S&P 500", os.path.join(DATA_DIR, "SP500.csv"), 1, 1),
]

SEQ_LEN = 30
BATCH_SIZE = 32
D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 2
DROPOUT = 0.10
EPOCHS = 50
PATIENCE = 6
VAL_RATIO = 0.1
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Realized-volatility protocol constants (Fairness_Audit_AllMethods_2026-04-08_v6).
# Daily stock/index: 7-day rolling window, 252 trading-day annualization.
ROLLING_WINDOW = 7
ANNUALIZATION = 252.0

VMD_K = 6
VMD_ALPHA = 3600.0
VMD_TAU = 0.0
VMD_TOL = 1e-7
VMD_MAX_ITER = 500
PE_ORDER = 3
PE_DELAY = 1
PE_THRESHOLD = 0.58

RECO_WEIGHT = 0.08
PERSISTENT_WEIGHT = 0.14
HIGHFREQ_WEIGHT = 0.04
NOISE_WEIGHT = 0.03
ROUTER_BALANCE_WEIGHT = 0.01
TOPO_PRED_WEIGHT = 0.03

TOPO_D = 3
TOPO_TAU = 1
TOPO_BINS = 15
TOPO_MAX_R = 2.0
TOPO_EMB_DIM = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TUNABLE_HYPERPARAMS = (
    "BATCH_SIZE",
    "D_MODEL",
    "NHEAD",
    "NUM_LAYERS",
    "DROPOUT",
    "EPOCHS",
    "PATIENCE",
    "VAL_RATIO",
    "LR",
    "WEIGHT_DECAY",
    "GRAD_CLIP",
    "VMD_K",
    "VMD_ALPHA",
    "VMD_TAU",
    "VMD_TOL",
    "VMD_MAX_ITER",
    "PE_ORDER",
    "PE_DELAY",
    "PE_THRESHOLD",
    "RECO_WEIGHT",
    "PERSISTENT_WEIGHT",
    "HIGHFREQ_WEIGHT",
    "NOISE_WEIGHT",
    "ROUTER_BALANCE_WEIGHT",
    "TOPO_PRED_WEIGHT",
    "TOPO_D",
    "TOPO_TAU",
    "TOPO_BINS",
    "TOPO_MAX_R",
    "TOPO_EMB_DIM",
)

BASE_HYPERPARAMS = {name: globals()[name] for name in TUNABLE_HYPERPARAMS}

DATASET_HYPERPARAM_PROFILES = {
    "AAPL": ("base_v6b", {}),
    "Amazon": (
        "lr2e4_drop005_auxlow",
        {
            "EPOCHS": 70,
            "PATIENCE": 10,
            "LR": 2e-4,
            "WEIGHT_DECAY": 5e-5,
            "DROPOUT": 0.05,
            "RECO_WEIGHT": 0.06,
            "PERSISTENT_WEIGHT": 0.10,
            "HIGHFREQ_WEIGHT": 0.02,
            "NOISE_WEIGHT": 0.02,
            "ROUTER_BALANCE_WEIGHT": 0.004,
            "TOPO_PRED_WEIGHT": 0.02,
        },
    ),
    "Google": (
        "layers3_h4_lr2e4_drop005_auxlow",
        {
            "NHEAD": 4,
            "NUM_LAYERS": 3,
            "EPOCHS": 70,
            "PATIENCE": 10,
            "LR": 2e-4,
            "WEIGHT_DECAY": 5e-5,
            "DROPOUT": 0.05,
            "RECO_WEIGHT": 0.06,
            "PERSISTENT_WEIGHT": 0.10,
            "HIGHFREQ_WEIGHT": 0.02,
            "NOISE_WEIGHT": 0.02,
            "ROUTER_BALANCE_WEIGHT": 0.004,
            "TOPO_PRED_WEIGHT": 0.02,
        },
    ),
    "Nasdaq": (
        "lr2e4_drop005_auxlow",
        {
            "EPOCHS": 70,
            "PATIENCE": 10,
            "LR": 2e-4,
            "WEIGHT_DECAY": 5e-5,
            "DROPOUT": 0.05,
            "RECO_WEIGHT": 0.06,
            "PERSISTENT_WEIGHT": 0.10,
            "HIGHFREQ_WEIGHT": 0.02,
            "NOISE_WEIGHT": 0.02,
            "ROUTER_BALANCE_WEIGHT": 0.004,
            "TOPO_PRED_WEIGHT": 0.02,
        },
    ),
    "NYSE": (
        "layers3_h4_lr5e5",
        {
            "NHEAD": 4,
            "NUM_LAYERS": 3,
            "EPOCHS": 70,
            "PATIENCE": 10,
            "LR": 5e-5,
        },
    ),
}


def apply_hyperparams_for_dataset(dataset_name):
    for key, value in BASE_HYPERPARAMS.items():
        globals()[key] = value

    profile_name, overrides = DATASET_HYPERPARAM_PROFILES.get(dataset_name, ("base_v6b", {}))
    for key, value in overrides.items():
        globals()[key] = value
    return profile_name


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_process_data(spec: DatasetSpec):
    df = pd.read_csv(spec.path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    else:
        raise ValueError(f"{spec.name}: this stock/index script expects a daily dataset with a Date column.")

    close_col = "Close" if "Close" in df.columns else "close"
    df["log_ret"] = np.log(df[close_col] / df[close_col].shift(1))
    df["rv"] = df["log_ret"].rolling(window=ROLLING_WINDOW).var() * ANNUALIZATION
    df = df.dropna()
    df = df[df["rv"] > 1e-8].copy()
    df["log_rv"] = np.log(df["rv"])
    df["log_vol"] = np.sqrt(df["rv"])
    return df[["log_rv", "log_vol"]].dropna().astype(np.float32)


def create_sequences(data, index_values, seq_len, stride=1):
    xs, ys, ts = [], [], []
    for i in range(0, len(data) - seq_len, stride):
        xs.append(data[i : i + seq_len])
        ys.append(data[i + seq_len, 0])
        ts.append(index_values[i + seq_len])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(ts)


def encode_daily_cyclical(ts):
    ts = pd.Timestamp(ts)
    day_of_week = ts.dayofweek
    month_of_year = ts.month - 1
    return np.asarray(
        [
            math.sin(2.0 * math.pi * day_of_week / 7.0),
            math.cos(2.0 * math.pi * day_of_week / 7.0),
            math.sin(2.0 * math.pi * month_of_year / 12.0),
            math.cos(2.0 * math.pi * month_of_year / 12.0),
        ],
        dtype=np.float32,
    )


def create_time_features(index_like, seq_len, stride=1):
    index = pd.DatetimeIndex(index_like)
    feats = []
    for i in range(0, len(index) - seq_len, stride):
        feats.append(encode_daily_cyclical(index[i + seq_len]))
    return np.asarray(feats, dtype=np.float32)


def compute_topo_features(x_array, d=3, tau=1, bins=15, max_r=2.0):
    r_vals = np.linspace(0, max_r, bins)
    out_dim = bins * 2
    topo_features = np.zeros((len(x_array), out_dim), dtype=np.float32)

    for idx, seq in enumerate(x_array):
        series = seq[:, 0].astype(np.float64)
        n_pts = len(series) - (d - 1) * tau
        if n_pts <= 0:
            continue

        point_cloud = np.array([series[j : j + d * tau : tau] for j in range(n_pts)], dtype=np.float64)
        try:
            dgms = ripser.ripser(point_cloud, maxdim=1)["dgms"]
            h0, h1 = dgms[0], dgms[1]
            h0_finite = h0[h0[:, 1] != np.inf]
            betti0 = np.zeros(bins, dtype=np.float64)
            betti1 = np.zeros(bins, dtype=np.float64)
            for b_idx, r in enumerate(r_vals):
                if len(h0_finite) > 0:
                    betti0[b_idx] = np.sum((h0_finite[:, 0] <= r) & (h0_finite[:, 1] > r)) / n_pts
                if len(h1) > 0:
                    betti1[b_idx] = np.sum((h1[:, 0] <= r) & (h1[:, 1] > r)) / n_pts
            topo_features[idx] = np.concatenate([betti0, betti1]).astype(np.float32)
        except Exception:
            topo_features[idx] = 0.0

        if (idx + 1) % 5000 == 0:
            print(f"    topo feature build [{idx + 1}/{len(x_array)}]")

    return topo_features


def permutation_entropy(signal, order=3, delay=1):
    n = len(signal)
    span = delay * (order - 1) + 1
    if n < span:
        return 1.0

    patterns = {}
    for i in range(n - span + 1):
        window = signal[i : i + span : delay]
        pattern = tuple(np.argsort(window))
        patterns[pattern] = patterns.get(pattern, 0) + 1

    counts = np.asarray(list(patterns.values()), dtype=np.float64)
    probs = counts / max(counts.sum(), 1.0)
    pe = -np.sum(probs * np.log(probs + 1e-12))
    return float(pe / np.log(math.factorial(order)))


def vmd_decompose_1d(signal, alpha, tau, k_modes, tol, max_iter):
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) % 2 == 1:
        signal = signal[:-1]

    half = len(signal) // 2
    mirrored = np.concatenate([signal[:half][::-1], signal, signal[-half:][::-1]])
    t_len = len(mirrored)

    freqs = (np.arange(1, t_len + 1) / t_len) - 0.5 - (1.0 / t_len)
    f_hat = np.fft.fftshift(np.fft.fft(mirrored))
    f_hat_plus = f_hat.copy()
    f_hat_plus[: t_len // 2] = 0

    u_hat_plus = np.zeros((max_iter, t_len, k_modes), dtype=np.complex128)
    omega_plus = np.zeros((max_iter, k_modes), dtype=np.float64)
    lambda_hat = np.zeros((max_iter, t_len), dtype=np.complex128)

    for k in range(k_modes):
        omega_plus[0, k] = 0.5 * k / max(k_modes, 1)

    u_diff = tol + np.finfo(float).eps
    iteration = 0
    sum_uk = np.zeros(t_len, dtype=np.complex128)

    while u_diff > tol and iteration < max_iter - 1:
        sum_uk = u_hat_plus[iteration, :, k_modes - 1] + sum_uk - u_hat_plus[iteration, :, 0]
        denom = 1.0 + alpha * (freqs - omega_plus[iteration, 0]) ** 2
        u_hat_plus[iteration + 1, :, 0] = (f_hat_plus - sum_uk - lambda_hat[iteration, :] / 2.0) / denom

        power0 = np.abs(u_hat_plus[iteration + 1, t_len // 2 :, 0]) ** 2
        if power0.sum() > 1e-12:
            omega_plus[iteration + 1, 0] = np.dot(freqs[t_len // 2 :], power0) / power0.sum()

        for k in range(1, k_modes):
            sum_uk = u_hat_plus[iteration + 1, :, k - 1] + sum_uk - u_hat_plus[iteration, :, k]
            denom = 1.0 + alpha * (freqs - omega_plus[iteration, k]) ** 2
            u_hat_plus[iteration + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[iteration, :] / 2.0) / denom

            powerk = np.abs(u_hat_plus[iteration + 1, t_len // 2 :, k]) ** 2
            if powerk.sum() > 1e-12:
                omega_plus[iteration + 1, k] = np.dot(freqs[t_len // 2 :], powerk) / powerk.sum()

        lambda_hat[iteration + 1, :] = lambda_hat[iteration, :] + tau * (
            np.sum(u_hat_plus[iteration + 1, :, :], axis=1) - f_hat_plus
        )

        iteration += 1
        u_diff = np.finfo(float).eps
        for k in range(k_modes):
            delta = u_hat_plus[iteration, :, k] - u_hat_plus[iteration - 1, :, k]
            u_diff += (1.0 / t_len) * np.dot(delta, np.conj(delta)).real

    final_idx = iteration
    u_hat = np.zeros((t_len, k_modes), dtype=np.complex128)
    u_hat[t_len // 2 :, :] = u_hat_plus[final_idx, t_len // 2 :, :]
    mirrored_idx = np.flip(np.arange(1, t_len // 2 + 1), axis=0)
    u_hat[mirrored_idx, :] = np.conj(u_hat_plus[final_idx, t_len // 2 :, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])

    modes = np.zeros((k_modes, t_len), dtype=np.float64)
    for k in range(k_modes):
        modes[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    return modes[:, t_len // 4 : 3 * t_len // 4]


def build_three_step_targets(x_seq):
    reco_targets = np.zeros((len(x_seq), SEQ_LEN), dtype=np.float32)
    persistent_targets = np.zeros((len(x_seq), SEQ_LEN), dtype=np.float32)
    highfreq_targets = np.zeros((len(x_seq), SEQ_LEN), dtype=np.float32)
    noise_targets = np.zeros((len(x_seq), SEQ_LEN), dtype=np.float32)
    state_features = np.zeros((len(x_seq), 2), dtype=np.float32)
    kept_counts = []
    persistent_counts = []

    for idx, seq in enumerate(x_seq):
        target_series = seq[:, 0].astype(np.float64)
        raw_entropy = permutation_entropy(target_series, order=PE_ORDER, delay=PE_DELAY)
        raw_volatility = float(np.std(target_series))
        modes = vmd_decompose_1d(
            target_series,
            alpha=VMD_ALPHA,
            tau=VMD_TAU,
            k_modes=VMD_K,
            tol=VMD_TOL,
            max_iter=VMD_MAX_ITER,
        )
        entropies = np.asarray(
            [permutation_entropy(mode, order=PE_ORDER, delay=PE_DELAY) for mode in modes],
            dtype=np.float64,
        )
        keep_mask = entropies <= PE_THRESHOLD
        if not keep_mask.any():
            keep_mask[np.argmin(entropies)] = True

        reco = modes[keep_mask].sum(axis=0).astype(np.float32)
        center_freq = np.asarray(
            [np.argmax(np.abs(np.fft.rfft(mode)) ** 2) for mode in modes],
            dtype=np.float64,
        )

        persistent_mask = np.zeros_like(keep_mask, dtype=bool)
        if keep_mask.any():
            kept_freqs = center_freq[keep_mask]
            persistent_mask = keep_mask & (center_freq <= np.median(kept_freqs))
            if not persistent_mask.any():
                kept_indices = np.where(keep_mask)[0]
                persistent_mask[kept_indices[np.argmin(center_freq[keep_mask])]] = True

        highfreq_mask = keep_mask & (~persistent_mask)
        noise_mask = ~keep_mask

        persistent = (
            modes[persistent_mask].sum(axis=0).astype(np.float32)
            if persistent_mask.any()
            else np.zeros(SEQ_LEN, dtype=np.float32)
        )
        highfreq = (
            modes[highfreq_mask].sum(axis=0).astype(np.float32)
            if highfreq_mask.any()
            else np.zeros(SEQ_LEN, dtype=np.float32)
        )
        noise = (
            modes[noise_mask].sum(axis=0).astype(np.float32)
            if noise_mask.any()
            else np.zeros(SEQ_LEN, dtype=np.float32)
        )

        reco_targets[idx] = reco
        persistent_targets[idx] = persistent
        highfreq_targets[idx] = highfreq
        noise_targets[idx] = noise
        state_features[idx] = np.asarray([raw_volatility, raw_entropy], dtype=np.float32)
        kept_counts.append(int(keep_mask.sum()))
        persistent_counts.append(int(persistent_mask.sum()))

    stats = {
        "avg_kept_imfs": float(np.mean(kept_counts)),
        "avg_persistent_imfs": float(np.mean(persistent_counts)),
    }
    return reco_targets, persistent_targets, highfreq_targets, noise_targets, state_features, stats


# Fixed white-box Transformer backbone












# ---------------------------------------------------------------------------
# Enhanced HOW3 fusion mechanism: element-wise gating with topo-informed routing
# ---------------------------------------------------------------------------




def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    return mse, rmse, mae, r2, mape


def train_how123(spec, x_train, y_train, t_train, x_val, y_val, t_val, x_test, y_test, t_test, scaler):
    topo_train = compute_topo_features(x_train, d=TOPO_D, tau=TOPO_TAU, bins=TOPO_BINS, max_r=TOPO_MAX_R)
    topo_val = compute_topo_features(x_val, d=TOPO_D, tau=TOPO_TAU, bins=TOPO_BINS, max_r=TOPO_MAX_R)
    topo_test = compute_topo_features(x_test, d=TOPO_D, tau=TOPO_TAU, bins=TOPO_BINS, max_r=TOPO_MAX_R)

    train_targets = build_three_step_targets(x_train)
    val_targets = build_three_step_targets(x_val)
    test_targets = build_three_step_targets(x_test)

    reco_train, persistent_train, highfreq_train, noise_train, state_train, train_stats = train_targets
    reco_val, persistent_val, highfreq_val, noise_val, state_val, _ = val_targets
    _, persistent_test, _, _, state_test, _ = test_targets

    persistent_input_train = x_train.copy()
    persistent_input_val = x_val.copy()
    persistent_input_test = x_test.copy()
    persistent_input_train[:, :, 0] = persistent_train
    persistent_input_val[:, :, 0] = persistent_val
    persistent_input_test[:, :, 0] = persistent_test

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(x_train),
            torch.FloatTensor(persistent_input_train),
            torch.FloatTensor(topo_train),
            torch.FloatTensor(state_train),
            torch.FloatTensor(t_train),
            torch.FloatTensor(y_train).unsqueeze(1),
            torch.FloatTensor(reco_train),
            torch.FloatTensor(persistent_train),
            torch.FloatTensor(highfreq_train),
            torch.FloatTensor(noise_train),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(x_val),
            torch.FloatTensor(persistent_input_val),
            torch.FloatTensor(topo_val),
            torch.FloatTensor(state_val),
            torch.FloatTensor(t_val),
            torch.FloatTensor(y_val).unsqueeze(1),
            torch.FloatTensor(reco_val),
            torch.FloatTensor(persistent_val),
            torch.FloatTensor(highfreq_val),
            torch.FloatTensor(noise_val),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = EXACT(
        channels=2, d_model=D_MODEL, seq_len=SEQ_LEN, nhead=NHEAD,
        num_layers=NUM_LAYERS, dropout=DROPOUT, topo_bins=TOPO_BINS,
        topo_emb_dim=TOPO_EMB_DIM, router_dropout=0.1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse_loss = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history_rows = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for raw_x, persistent_x, topo_batch, state_batch, time_batch, y_batch, reco_batch, persistent_batch, highfreq_batch, noise_batch in train_loader:
            raw_x = raw_x.to(device)
            persistent_x = persistent_x.to(device)
            topo_batch = topo_batch.to(device)
            state_batch = state_batch.to(device)
            time_batch = time_batch.to(device)
            y_batch = y_batch.to(device)
            reco_batch = reco_batch.to(device)
            persistent_batch = persistent_batch.to(device)
            highfreq_batch = highfreq_batch.to(device)
            noise_batch = noise_batch.to(device)

            optimizer.zero_grad()
            pred, gate, reco_hat, persistent_hat, highfreq_hat, noise_hat, topo_pred = model(
                raw_x, persistent_x, topo_batch, state_batch, time_batch
            )
            loss_pred = mse_loss(pred, y_batch)
            loss_reco = mse_loss(reco_hat, reco_batch)
            loss_persistent = mse_loss(persistent_hat, persistent_batch)
            loss_highfreq = mse_loss(highfreq_hat, highfreq_batch)
            loss_noise = mse_loss(noise_hat, noise_batch)
            loss_balance = (gate.mean() - 0.5).pow(2)
            loss_topo = mse_loss(topo_pred, topo_batch)
            loss = (
                loss_pred
                + RECO_WEIGHT * loss_reco
                + PERSISTENT_WEIGHT * loss_persistent
                + HIGHFREQ_WEIGHT * loss_highfreq
                + NOISE_WEIGHT * loss_noise
                + ROUTER_BALANCE_WEIGHT * loss_balance
                + TOPO_PRED_WEIGHT * loss_topo
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for raw_x, persistent_x, topo_batch, state_batch, time_batch, y_batch, reco_batch, persistent_batch, highfreq_batch, noise_batch in val_loader:
                raw_x = raw_x.to(device)
                persistent_x = persistent_x.to(device)
                topo_batch = topo_batch.to(device)
                state_batch = state_batch.to(device)
                time_batch = time_batch.to(device)
                y_batch = y_batch.to(device)
                reco_batch = reco_batch.to(device)
                persistent_batch = persistent_batch.to(device)
                highfreq_batch = highfreq_batch.to(device)
                noise_batch = noise_batch.to(device)
                pred, gate, reco_hat, persistent_hat, highfreq_hat, noise_hat, topo_pred = model(
                    raw_x, persistent_x, topo_batch, state_batch, time_batch
                )
                loss_pred = mse_loss(pred, y_batch)
                loss_reco = mse_loss(reco_hat, reco_batch)
                loss_persistent = mse_loss(persistent_hat, persistent_batch)
                loss_highfreq = mse_loss(highfreq_hat, highfreq_batch)
                loss_noise = mse_loss(noise_hat, noise_batch)
                loss_balance = (gate.mean() - 0.5).pow(2)
                loss_topo = mse_loss(topo_pred, topo_batch)
                val_loss += (
                    loss_pred
                    + RECO_WEIGHT * loss_reco
                    + PERSISTENT_WEIGHT * loss_persistent
                    + HIGHFREQ_WEIGHT * loss_highfreq
                    + NOISE_WEIGHT * loss_noise
                    + ROUTER_BALANCE_WEIGHT * loss_balance
                    + TOPO_PRED_WEIGHT * loss_topo
                ).item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        history_rows.append(
            {"Model": "TDC", "Dataset": spec.name, "Epoch": epoch + 1, "Train_Loss": train_loss, "Val_Loss": val_loss}
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    test_x = torch.FloatTensor(x_test).to(device)
    test_p = torch.FloatTensor(persistent_input_test).to(device)
    test_topo = torch.FloatTensor(topo_test).to(device)
    test_state = torch.FloatTensor(state_test).to(device)
    test_time = torch.FloatTensor(t_test).to(device)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p, g, _, _, _, _, _ = model(test_x, test_p, test_topo, test_state, test_time)
        pred_scaled = p.cpu().numpy().reshape(-1)
        avg_gate_fine = float(g.mean().cpu().numpy())

    rv_scale = float(scaler.scale_[0])
    rv_mean = float(scaler.mean_[0])
    pred_log_rv = pred_scaled * rv_scale + rv_mean
    true_log_rv = y_test * rv_scale + rv_mean
    mse, rmse, mae, r2, mape = calculate_metrics(true_log_rv, pred_log_rv)

    return {
        "Model": "TDC",
        "Dataset": spec.name,
        "Hyperparam_Profile": DATASET_HYPERPARAM_PROFILES.get(spec.name, ("base_v6b", {}))[0],
        "Train_Stride": spec.train_stride,
        "Test_Stride": spec.test_stride,
        "Best_Epoch": best_epoch,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE_pct": mape,
        "Router_Fine": avg_gate_fine,
        "Router_Coarse": 1.0 - avg_gate_fine,
        "Avg_Kept_IMFs": train_stats["avg_kept_imfs"],
        "Avg_Persistent_IMFs": train_stats["avg_persistent_imfs"],
        "Time_Feature_Mode": "daily_calendar",
        "Target_Protocol": "daily_rolling_rv",
    }, history_rows


def run_dataset(spec):
    df = load_and_process_data(spec)
    split_idx = int(len(df) * 0.8)
    train_raw = df.iloc[:split_idx].copy()
    test_raw = df.iloc[split_idx:].copy()

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
    test_scaled = scaler.transform(test_raw).astype(np.float32)

    x_train_full, y_train_full, _ = create_sequences(train_scaled, train_raw.index, SEQ_LEN, stride=spec.train_stride)
    t_train_full = create_time_features(train_raw.index, SEQ_LEN, stride=spec.train_stride)
    x_test, y_test, _ = create_sequences(test_scaled, test_raw.index, SEQ_LEN, stride=spec.test_stride)
    t_test = create_time_features(test_raw.index, SEQ_LEN, stride=spec.test_stride)

    val_size = max(1, int(len(x_train_full) * VAL_RATIO))
    x_train, x_val = x_train_full[:-val_size], x_train_full[-val_size:]
    y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]
    t_train, t_val = t_train_full[:-val_size], t_train_full[-val_size:]

    how123_summary, how123_history = train_how123(
        spec, x_train, y_train, t_train, x_val, y_val, t_val, x_test, y_test, t_test, scaler
    )
    return [how123_summary], how123_history


def format_float_columns(df, exclude=None):
    exclude = set(exclude or [])
    formatted = df.copy()
    for col in formatted.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: f"{float(x):.4f}")
    return formatted
