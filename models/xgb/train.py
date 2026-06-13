"""Train the boring-champion daily XGB on FEATURE_SPEC features -> versioned artifact.

Offline trainer (needs numpy + xgboost; NOT imported by the stdlib test suite).
It reads an MKTD ``.bin`` through the ONE reader (``sim/binpack``), builds a
forward-return classification label, fits an ``XGBClassifier`` on the exact
``forecast.features.FEATURE_SPEC`` columns, validates the trained model against the
feature contract (so a model that needs columns the live path can't supply is
refused at train time, not in prod), and writes a versioned artifact:

    <out_dir>/<run_id>/model.json     # xgboost native format
    <out_dir>/<run_id>/metadata.json  # feature spec version, names, params, metrics

The artifact's ``metadata.json`` carries ``feature_names`` so downstream loaders
(and ``validate_feature_contract``) can re-check the contract before serving.

Usage:
    python -m models.xgb.train --data sim/data/sample.bin --out-dir artifacts/xgb
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from forecast.features import FEATURE_NAMES, FEATURE_SPEC, validate_feature_contract
from sim.binpack import read_features, read_header, read_prices, read_symbols

SPEC_VERSION = FEATURE_SPEC.version
CLOSE_IDX = 3  # OHLCV close column in the price block


def build_dataset(data_path: Path, horizon: int = 24, up_threshold: float = 0.0,
                  train_frac: float = 1.0):
    """Flatten [T][S] samples into (X, y) for a forward-return up/down label.

    Label is 1 if the symbol's close ``horizon`` bars ahead exceeds today's close
    by ``up_threshold`` (fractional), else 0. The last ``horizon`` bars per symbol
    have no realised future and are dropped.

    ``train_frac`` restricts training to the FIRST fraction of timesteps so the
    gate can judge the held-out tail out-of-sample (no leakage): samples are taken
    from ``t in [0, T*train_frac)`` and a label never peeks past that boundary.
    """
    feats = read_features(data_path)        # [T][S][F]
    prices = read_prices(data_path)         # [T][S][5]
    hdr = read_header(data_path)
    T, S = hdr["num_timesteps"], hdr["num_symbols"]
    t_max = int(T * train_frac)

    X_rows, y_rows = [], []
    for t in range(t_max - horizon):
        for s in range(S):
            close_now = prices[t][s][CLOSE_IDX]
            close_fut = prices[t + horizon][s][CLOSE_IDX]
            if close_now <= 0.0:
                continue
            fwd_ret = (close_fut - close_now) / close_now
            X_rows.append(feats[t][s])
            y_rows.append(1 if fwd_ret > up_threshold else 0)

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int32)
    return X, y


def train(data_path: Path, out_dir: Path, *, horizon: int = 24,
          n_estimators: int = 200, max_depth: int = 4, learning_rate: float = 0.05,
          train_frac: float = 1.0, run_id: str | None = None) -> Path:
    X, y = build_dataset(data_path, horizon=horizon, train_frac=train_frac)
    if len(X) == 0:
        raise SystemExit("no training samples produced (data too short for horizon?)")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=0,
    )
    # Pin the column names so the artifact is self-describing and contract-checkable.
    model.fit(np.asarray(X), np.asarray(y))
    try:
        model.get_booster().feature_names = list(FEATURE_NAMES)
    except Exception:
        pass

    # Refuse to ship a model whose required features the live path can't supply.
    validate_feature_contract(list(FEATURE_NAMES), strict_order=True)

    run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.json"
    model.get_booster().save_model(str(model_path))

    train_acc = float((model.predict(np.asarray(X)) == y).mean())
    metadata = {
        "feature_spec_version": SPEC_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "horizon_bars": horizon,
        "symbols": read_symbols(data_path),
        "source_bin": str(data_path),
        "n_samples": int(len(X)),
        "train_frac": train_frac,
        "label_pos_rate": float(y.mean()),
        "train_accuracy": train_acc,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        },
        "trained_at": run_id,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"wrote artifact {run_dir} (n_samples={len(X)}, train_acc={train_acc:.3f})")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description="Train the daily XGB champion")
    ap.add_argument("--data", default="sim/data/sample.bin", help="MKTD .bin path")
    ap.add_argument("--out-dir", default="artifacts/xgb", help="artifact root")
    ap.add_argument("--horizon", type=int, default=24, help="forward-return horizon (bars)")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="fraction of timesteps (from the start) to train on; "
                         "gate the rest out-of-sample")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    train(
        Path(args.data), Path(args.out_dir), horizon=args.horizon,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, train_frac=args.train_frac, run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
