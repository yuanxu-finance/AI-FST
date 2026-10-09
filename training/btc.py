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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import ripser
except ImportError as exc:
    raise ImportError(
        "Missing dependency: 'ripser'. Install via 'pip install ripser'."
    ) from exc


OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "outputs")
DATA_DIR = str(Path(__file__).resolve().parents[1] / "datasets")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str


DATASETS = [
    DatasetSpec("BTC", os.path.join(DATA_DIR, "BTC.csv")),
]

SEQ_LEN = 30
SEQ_STRIDE = 60
ROLLING_WINDOW = 7
ANNUALIZATION = 365.0 * 24.0 * 60.0
BATCH_SIZE = 128
D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 2
DROPOUT = 0.10
EPOCHS = 20
PATIENCE = 4
VAL_RATIO = 0.1
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

VMD_K = 6
VMD_ALPHA = 3600.0
VMD_TAU = 0.0
VMD_TOL = 1e-7
VMD_MAX_ITER = 500
PE_ORDER = 3
PE_DELAY = 1
PE_THRESHOLD = 0.58

RECO_WEIGHT = 0.08
PERSISTENT_WEIGHT = 0.12
HIGHFREQ_WEIGHT = 0.04
NOISE_WEIGHT = 0.03
ROUTER_BALANCE_WEIGHT = 0.003
TOPO_PRED_WEIGHT = 0.03

TOPO_D = 3
TOPO_TAU = 1
TOPO_BINS = 10
TOPO_MAX_R = 2.0
TOPO_EMB_DIM = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_process_data(filepath):
    df = pd.read_csv(filepath)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    else:
        raise ValueError(f"{filepath}: no Date/timestamp column found.")

    close_col = "Close" if "Close" in df.columns else "close"
    close = df[close_col].astype(float)
    df["log_ret"] = np.log(close / close.shift(1))
    df["rv"] = df["log_ret"].rolling(window=ROLLING_WINDOW).var() * ANNUALIZATION
    df = df.dropna()
    df = df[df["rv"] > 1e-8].copy()
    df["log_rv"] = np.log(df["rv"])
    df["log_vol"] = np.sqrt(df["rv"])
    return df[["log_rv", "log_vol"]].dropna().astype(np.float32)


def create_sequences(data, index_values, seq_len, stride):
    xs, ys, ts = [], [], []
    for i in range(0, len(data) - seq_len, stride):
        xs.append(data[i : i + seq_len])
        ys.append(data[i + seq_len, 0])
        ts.append(index_values[i + seq_len])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(ts),
    )


def encode_timestamp_cyclical(ts):
    ts = pd.Timestamp(ts)
    minute_of_day = ts.hour * 60 + ts.minute
    day_of_week = ts.dayofweek
    return np.asarray(
        [
            math.sin(2.0 * math.pi * minute_of_day / 1440.0),
            math.cos(2.0 * math.pi * minute_of_day / 1440.0),
            math.sin(2.0 * math.pi * day_of_week / 7.0),
            math.cos(2.0 * math.pi * day_of_week / 7.0),
        ],
        dtype=np.float32,
    )


def create_time_features(index_like, seq_len, stride):
    index = pd.DatetimeIndex(index_like)
    feats = []
    for i in range(0, len(index) - seq_len, stride):
        feats.append(encode_timestamp_cyclical(index[i + seq_len]))
    return np.asarray(feats, dtype=np.float32)


def compute_topo_features(
    x_array, d=TOPO_D, tau=TOPO_TAU, bins=TOPO_BINS, max_r=TOPO_MAX_R
):
    r_vals = np.linspace(0.0, max_r, bins)
    out_dim = bins * 2
    topo_features = np.zeros((len(x_array), out_dim), dtype=np.float32)

    for idx, seq in enumerate(x_array):
        series = seq[:, 0].astype(np.float64)
        n_pts = len(series) - (d - 1) * tau
        if n_pts <= 0:
            continue

        point_cloud = np.array(
            [series[j : j + d * tau : tau] for j in range(n_pts)], dtype=np.float64
        )
        try:
            dgms = ripser.ripser(point_cloud, maxdim=1)["dgms"]
            h0, h1 = dgms[0], dgms[1]
            h0_finite = h0[h0[:, 1] != np.inf]
            betti0 = np.zeros(bins, dtype=np.float64)
            betti1 = np.zeros(bins, dtype=np.float64)
            for b_idx, r in enumerate(r_vals):
                if len(h0_finite) > 0:
                    betti0[b_idx] = (
                        np.sum((h0_finite[:, 0] <= r) & (h0_finite[:, 1] > r)) / n_pts
                    )
                if len(h1) > 0:
                    betti1[b_idx] = np.sum((h1[:, 0] <= r) & (h1[:, 1] > r)) / n_pts
            topo_features[idx] = np.concatenate([betti0, betti1]).astype(np.float32)
        except Exception:
            topo_features[idx] = 0.0

        if (idx + 1) % 5000 == 0 or (idx + 1) == len(x_array):
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
        sum_uk = (
            u_hat_plus[iteration, :, k_modes - 1] + sum_uk - u_hat_plus[iteration, :, 0]
        )
        denom = 1.0 + alpha * (freqs - omega_plus[iteration, 0]) ** 2
        u_hat_plus[iteration + 1, :, 0] = (
            f_hat_plus - sum_uk - lambda_hat[iteration, :] / 2.0
        ) / denom

        power0 = np.abs(u_hat_plus[iteration + 1, t_len // 2 :, 0]) ** 2
        if power0.sum() > 1e-12:
            omega_plus[iteration + 1, 0] = (
                np.dot(freqs[t_len // 2 :], power0) / power0.sum()
            )

        for k in range(1, k_modes):
            sum_uk = (
                u_hat_plus[iteration + 1, :, k - 1]
                + sum_uk
                - u_hat_plus[iteration, :, k]
            )
            denom = 1.0 + alpha * (freqs - omega_plus[iteration, k]) ** 2
            u_hat_plus[iteration + 1, :, k] = (
                f_hat_plus - sum_uk - lambda_hat[iteration, :] / 2.0
            ) / denom

            powerk = np.abs(u_hat_plus[iteration + 1, t_len // 2 :, k]) ** 2
            if powerk.sum() > 1e-12:
                omega_plus[iteration + 1, k] = (
                    np.dot(freqs[t_len // 2 :], powerk) / powerk.sum()
                )

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
            [
                permutation_entropy(mode, order=PE_ORDER, delay=PE_DELAY)
                for mode in modes
            ],
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
        state_features[idx] = np.asarray(
            [raw_volatility, raw_entropy], dtype=np.float32
        )
        kept_counts.append(int(keep_mask.sum()))
        persistent_counts.append(int(persistent_mask.sum()))

    stats = {
        "avg_kept_imfs": float(np.mean(kept_counts)),
        "avg_persistent_imfs": float(np.mean(persistent_counts)),
    }
    return (
        reco_targets,
        persistent_targets,
        highfreq_targets,
        noise_targets,
        state_features,
        stats,
    )


def inverse_target(scaler, values):
    return values * float(scaler.scale_[0]) + float(scaler.mean_[0])


def safe_pearsonr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def calculate_metrics(true_scaled, pred_scaled, last_input_scaled, scaler):
    true_log_rv = inverse_target(scaler, true_scaled.reshape(-1))
    pred_log_rv = inverse_target(scaler, pred_scaled.reshape(-1))
    last_log_rv = inverse_target(scaler, last_input_scaled.reshape(-1))

    true_rv = np.exp(true_log_rv)
    pred_rv = np.exp(pred_log_rv)

    mse = mean_squared_error(true_log_rv, pred_log_rv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_log_rv, pred_log_rv)
    r2 = r2_score(true_log_rv, pred_log_rv)
    mask = np.abs(true_log_rv) > 1e-8
    mape = (
        np.mean(np.abs((true_log_rv[mask] - pred_log_rv[mask]) / true_log_rv[mask]))
        * 100.0
    )

    true_dir = np.sign(true_log_rv - last_log_rv)
    pred_dir = np.sign(pred_log_rv - last_log_rv)
    accuracy = float(np.mean(true_dir == pred_dir) * 100.0)

    eps = 1e-12
    ratio = true_rv / np.maximum(pred_rv, eps)
    qlike = float(np.mean(ratio - np.log(ratio + eps) - 1.0))
    smape_rv = float(
        np.mean(
            200.0
            * np.abs(pred_rv - true_rv)
            / (np.abs(true_rv) + np.abs(pred_rv) + eps)
        )
    )
    pearson_r = safe_pearsonr(true_log_rv, pred_log_rv)

    prediction_df = pd.DataFrame(
        {
            "True_Log_RV": true_log_rv,
            "Pred_Log_RV": pred_log_rv,
            "True_RV": true_rv,
            "Pred_RV": pred_rv,
            "Last_Log_RV": last_log_rv,
            "True_Direction": true_dir,
            "Pred_Direction": pred_dir,
            "Direction_Correct": (true_dir == pred_dir).astype(int),
        }
    )
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "MAPE": float(mape),
        "Accuracy": accuracy,
        "QLIKE": qlike,
        "SMAPE_RV": smape_rv,
        "PearsonR": pearson_r,
    }, prediction_df


def train_exact(
    spec,
    x_train,
    y_train,
    t_train,
    x_val,
    y_val,
    t_val,
    x_test,
    y_test,
    t_test,
    test_ts,
    scaler,
):
    topo_train = compute_topo_features(x_train)
    topo_val = compute_topo_features(x_val)
    topo_test = compute_topo_features(x_test)

    train_targets = build_three_step_targets(x_train)
    val_targets = build_three_step_targets(x_val)
    test_targets = build_three_step_targets(x_test)

    (
        reco_train,
        persistent_train,
        highfreq_train,
        noise_train,
        state_train,
        train_stats,
    ) = train_targets
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
        channels=2,
        d_model=D_MODEL,
        seq_len=SEQ_LEN,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        topo_bins=TOPO_BINS,
        topo_emb_dim=TOPO_EMB_DIM,
        router_dropout=DROPOUT,
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
        for (
            raw_x,
            persistent_x,
            topo_x,
            state_x,
            time_x,
            y_batch,
            reco_batch,
            persistent_batch,
            highfreq_batch,
            noise_batch,
        ) in train_loader:
            raw_x = raw_x.to(device)
            persistent_x = persistent_x.to(device)
            topo_x = topo_x.to(device)
            state_x = state_x.to(device)
            time_x = time_x.to(device)
            y_batch = y_batch.to(device)
            reco_batch = reco_batch.to(device)
            persistent_batch = persistent_batch.to(device)
            highfreq_batch = highfreq_batch.to(device)
            noise_batch = noise_batch.to(device)

            optimizer.zero_grad()
            pred, gate, reco_hat, persistent_hat, highfreq_hat, noise_hat, topo_pred = (
                model(raw_x, persistent_x, topo_x, state_x, time_x)
            )
            loss_pred = mse_loss(pred, y_batch)
            loss_reco = mse_loss(reco_hat, reco_batch)
            loss_persistent = mse_loss(persistent_hat, persistent_batch)
            loss_highfreq = mse_loss(highfreq_hat, highfreq_batch)
            loss_noise = mse_loss(noise_hat, noise_batch)
            loss_balance = (gate.mean() - 0.5).pow(2)
            loss_topo = mse_loss(topo_pred, topo_x)
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
            for (
                raw_x,
                persistent_x,
                topo_x,
                state_x,
                time_x,
                y_batch,
                reco_batch,
                persistent_batch,
                highfreq_batch,
                noise_batch,
            ) in val_loader:
                raw_x = raw_x.to(device)
                persistent_x = persistent_x.to(device)
                topo_x = topo_x.to(device)
                state_x = state_x.to(device)
                time_x = time_x.to(device)
                y_batch = y_batch.to(device)
                reco_batch = reco_batch.to(device)
                persistent_batch = persistent_batch.to(device)
                highfreq_batch = highfreq_batch.to(device)
                noise_batch = noise_batch.to(device)
                (
                    pred,
                    gate,
                    reco_hat,
                    persistent_hat,
                    highfreq_hat,
                    noise_hat,
                    topo_pred,
                ) = model(raw_x, persistent_x, topo_x, state_x, time_x)
                val_loss += (
                    mse_loss(pred, y_batch)
                    + RECO_WEIGHT * mse_loss(reco_hat, reco_batch)
                    + PERSISTENT_WEIGHT * mse_loss(persistent_hat, persistent_batch)
                    + HIGHFREQ_WEIGHT * mse_loss(highfreq_hat, highfreq_batch)
                    + NOISE_WEIGHT * mse_loss(noise_hat, noise_batch)
                    + ROUTER_BALANCE_WEIGHT * (gate.mean() - 0.5).pow(2)
                    + TOPO_PRED_WEIGHT * mse_loss(topo_pred, topo_x)
                ).item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        history_rows.append(
            {
                "Dataset": spec.name,
                "Epoch": epoch + 1,
                "Train_Loss": train_loss,
                "Val_Loss": val_loss,
            }
        )
        print(
            f"{spec.name} | Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"{spec.name} | Early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred_scaled, gate, _, _, _, _, _ = model(
            torch.FloatTensor(x_test).to(device),
            torch.FloatTensor(persistent_input_test).to(device),
            torch.FloatTensor(topo_test).to(device),
            torch.FloatTensor(state_test).to(device),
            torch.FloatTensor(t_test).to(device),
        )
        pred_scaled = pred_scaled.cpu().numpy().reshape(-1)
        avg_gate = float(gate.mean().cpu().numpy())

    metrics, prediction_df = calculate_metrics(
        true_scaled=y_test,
        pred_scaled=pred_scaled,
        last_input_scaled=x_test[:, -1, 0],
        scaler=scaler,
    )
    prediction_df.insert(0, "Timestamp", pd.to_datetime(test_ts[: len(prediction_df)]))
    prediction_df.insert(0, "Dataset", spec.name)

    summary_row = {"Dataset": spec.name}
    summary_row.update(metrics)
    summary_row["Model"] = "EXACT"
    summary_row["Router_Fine"] = avg_gate
    summary_row["Router_Coarse"] = 1.0 - avg_gate
    summary_row["Best_Epoch"] = best_epoch
    summary_row["Avg_Kept_IMFs"] = train_stats["avg_kept_imfs"]
    summary_row["Avg_Persistent_IMFs"] = train_stats["avg_persistent_imfs"]
    return summary_row, history_rows, prediction_df


def run_dataset(spec):
    df = load_and_process_data(spec.path)
    split_idx = int(len(df) * 0.8)
    train_raw = df.iloc[:split_idx].copy()
    test_raw = df.iloc[split_idx:].copy()

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
    test_scaled = scaler.transform(test_raw).astype(np.float32)

    x_train_full, y_train_full, t_train_full_ts = create_sequences(
        train_scaled, train_raw.index, SEQ_LEN, stride=SEQ_STRIDE
    )
    x_test, y_test, t_test_ts = create_sequences(
        test_scaled, test_raw.index, SEQ_LEN, stride=SEQ_STRIDE
    )
    t_train_full = create_time_features(train_raw.index, SEQ_LEN, stride=SEQ_STRIDE)
    t_test = create_time_features(test_raw.index, SEQ_LEN, stride=SEQ_STRIDE)

    val_size = max(1, int(len(x_train_full) * VAL_RATIO))
    x_train, x_val = x_train_full[:-val_size], x_train_full[-val_size:]
    y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]
    t_train, t_val = t_train_full[:-val_size], t_train_full[-val_size:]

    summary_row, history_rows, prediction_df = train_exact(
        spec,
        x_train,
        y_train,
        t_train,
        x_val,
        y_val,
        t_val,
        x_test,
        y_test,
        t_test,
        t_test_ts,
        scaler,
    )
    return summary_row, history_rows, prediction_df


def format_float_columns(df, exclude=None):
    exclude = set() if exclude is None else set(exclude)
    out = df.copy()
    for col in out.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return out
