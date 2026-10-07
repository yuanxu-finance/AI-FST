"""Command-line entry point for EXACT financial time-series forecasting."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = {
    'AAPL': ('stock', 'AAPL'), 'Amazon': ('stock', 'Amazon'),
    'Google': ('stock', 'Google'), 'Nasdaq': ('stock', 'Nasdaq'),
    'NYSE': ('stock', 'NYSE'), 'CSI300': ('stock', 'CSI 300'),
    'SP500': ('stock', 'S&P 500'), 'BTC': ('btc', 'BTC'),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', choices=list(DATASETS) + ['all'])
    p.add_argument('--data_dir', type=Path, default=ROOT / 'datasets',
                   help='Root containing <dataset>/<dataset>.csv, or flat CSV files')
    p.add_argument('--output_dir', type=Path, default=ROOT / 'outputs')
    p.add_argument('--list_datasets', action='store_true')
    p.add_argument('--check_data', action='store_true', help='Validate data and split sizes without training')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = p.parse_args(argv)
    if not args.list_datasets and args.data is None:
        p.error('--data is required unless --list_datasets is used')
    for name in ['epochs', 'patience', 'batch_size', 'lr']:
        value = getattr(args, name)
        if value is not None and value <= 0:
            p.error(f'--{name} must be positive')
    if args.weight_decay is not None and args.weight_decay < 0:
        p.error('--weight_decay must be nonnegative')
    return args


def data_path(root, name):
    nested = root / name / f'{name}.csv'
    return nested if nested.is_file() else root / f'{name}.csv'


def validate_data(path, track, engine, spec):
    import numpy as np
    import pandas as pd
    raw = pd.read_csv(path)
    close = 'Close' if 'Close' in raw else 'close'
    if close not in raw:
        raise ValueError(f'{path}: requires Close or close')
    if track == 'btc' and 'timestamp' in raw:
        dates = pd.to_datetime(raw.timestamp, unit='s', errors='raise', utc=True)
    elif 'Date' in raw:
        dates = pd.to_datetime(raw.Date, errors='raise', utc=True)
    else:
        raise ValueError(f'{path}: requires Date (or timestamp for BTC)')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f'{path}: dates must be valid, unique, and increasing')
    if raw.isna().any().any():
        raise ValueError(f'{path}: missing values found; resolve them explicitly before training')
    price = pd.to_numeric(raw[close], errors='raise').to_numpy()
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError(f'{path}: close prices must be finite and positive')
    if track == 'btc' and not dates.diff().dropna().eq(pd.Timedelta(minutes=1)).all():
        raise ValueError(f'{path}: BTC track requires consecutive one-minute observations')
    cleaned = engine.load_and_process_data(spec if track == 'stock' else str(path))
    n = len(cleaned)
    cut = int(n * .8)
    stride = 1 if track == 'stock' else engine.SEQ_STRIDE
    count = lambda size: len(range(0, max(0, size-engine.SEQ_LEN), stride))
    train_full, test = count(cut), count(n-cut)
    val = max(1, int(train_full * engine.VAL_RATIO))
    if train_full-val < 2 or test < 2:
        raise ValueError(f'{path}: not enough windows for training/validation/testing')
    return {'file': str(path.resolve()), 'raw_rows':len(raw), 'cleaned_rows':n,
            'start_utc':str(dates.iloc[0]), 'end_utc':str(dates.iloc[-1]),
            'train_windows':train_full-val, 'validation_windows':val, 'test_windows':test}


def main(argv=None):
    args = parse_args(argv)
    if args.list_datasets:
        for name, (track, _) in DATASETS.items():
            available = data_path(args.data_dir, name).is_file()
            print(f'{name:8s} {track:5s} {"available" if available else "missing"}')
        return
    import torch
    import pandas as pd
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    names = list(DATASETS) if args.data == 'all' else [args.data]
    # Check all files before starting an expensive multi-dataset run.
    for name in names:
        if not data_path(args.data_dir, name).is_file():
            raise FileNotFoundError(data_path(args.data_dir, name))
    for name in names:
        track, original_name = DATASETS[name]
        engine = importlib.import_module('training.' + track)
        profile = engine.apply_hyperparams_for_dataset(original_name) if track == 'stock' else 'btc_original'
        for option, setting in [('epochs','EPOCHS'),('patience','PATIENCE'),('batch_size','BATCH_SIZE'),('lr','LR'),('weight_decay','WEIGHT_DECAY')]:
            if getattr(args, option) is not None:
                setattr(engine, setting, getattr(args, option))
        engine.device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
        engine.set_seed(args.seed)
        path = data_path(args.data_dir, name)
        spec = engine.DatasetSpec(original_name, str(path), 1, 1) if track == 'stock' else engine.DatasetSpec(original_name, str(path))
        audit = validate_data(path, track, engine, spec)
        print(json.dumps({'dataset':name, **audit}, indent=2))
        if args.check_data:
            continue
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        settings = {k:v for k,v in vars(engine).items() if k.isupper() and isinstance(v,(int,float,str,bool)) and k not in ['OUTPUT_DIR','DATA_DIR']}
        (out/'run_config.json').write_text(json.dumps({'dataset':name,'profile':profile,'seed':args.seed,'device':str(engine.device),'data':audit,'settings':settings},indent=2), encoding='utf-8')
        if track == 'stock':
            summaries, history = engine.run_dataset(spec)
            summary = pd.DataFrame(summaries)
        else:
            row, history, predictions = engine.run_dataset(spec)
            summary = pd.DataFrame([row])
            predictions.to_csv(out/'predictions.csv',index=False)
        summary.to_csv(out/'summary.csv',index=False)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(summary.to_string(index=False))
        print(f'Saved results: {out.resolve()}')


if __name__ == '__main__':
    main()
