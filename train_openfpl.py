#!/usr/bin/env python3
"""
OpenFPL — Train per-position ensembles (paper-aligned, simplified).

Inputs:
  - A *feature-ready* CSV with rows for many past GWs:
      season, gw, position, player, team, opponent, home, <numeric features...>, target
    where `target` is the realized FPL points for that GW.

What it does:
  1) Infers numeric feature columns (per position) and fits a *global* X scaler
  2) Fits a Y scaler on `target`
  3) For each position (GK, DEF, MID, FWD), runs GroupKFold CV with group=team
  4) Evaluates a small grid of RandomForestRegressor and, if available, XGBRegressor
  5) Selects the Top-K candidates per CV fold and saves them to:
        models_dir/cv{fold}_{POS}/{candidate_id}/model.pkl
     and writes a `search.txt` summary in each cv folder
  6) Saves artifacts in models_dir:
        xscaler.save, yscaler.save, features.save (dict: pos -> feature list)

After training, run:
  python predict_openfpl.py --samples data/samples_gwN.csv --models models_custom --out data/preds_gwN.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    XGBRegressor = None
    HAVE_XGB = False


META_COLS_DEFAULT = ["season", "gw", "position", "player", "team", "opponent", "home"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train OpenFPL-like per-position ensembles")
    ap.add_argument("--train", required=True, help="Path to feature-ready training CSV (many past GWs)")
    ap.add_argument("--models", required=True, help="Directory to write trained artifacts")
    ap.add_argument("--positions", default="GK,DEF,MID,FWD", help="Comma-separated list of positions to train")
    ap.add_argument("--target-col", default="target", help="Column name of realized FPL points")
    ap.add_argument("--meta-cols", default=",".join(META_COLS_DEFAULT),
                    help=f"Comma-separated meta columns to keep: defaults to {META_COLS_DEFAULT}")
    ap.add_argument("--top-k", type=int, default=10, help="Number of best candidates to keep per CV fold")
    ap.add_argument("--cv", type=int, default=5, help="Number of GroupKFold splits")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()


def select_feature_columns(df: pd.DataFrame, meta_cols: List[str], target_col: str) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in meta_cols + [target_col]:
        if c in num_cols:
            num_cols.remove(c)
    if not num_cols:
        raise ValueError("No numeric feature columns found. Ensure your CSV has engineered numeric features.")
    return num_cols


def candidate_grid() -> List[Tuple[str, dict]]:
    grid: List[Tuple[str, dict]] = []
    for n in [300, 600]:
        for depth in [None, 10, 16]:
            for leaf in [1, 3]:
                grid.append(("rf", {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "n_jobs": -1, "random_state": 0}))
    if HAVE_XGB:
        for n in [400, 800]:
            for depth in [4, 6, 8]:
                for lr in [0.05, 0.1]:
                    grid.append(("xgb", {
                        "n_estimators": n, "max_depth": depth, "learning_rate": lr,
                        "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0,
                        "random_state": 0, "n_jobs": -1, "tree_method": "hist"
                    }))
    return grid


def make_model(kind: str, params: dict):
    if kind == "rf":
        return RandomForestRegressor(**params)
    elif kind == "xgb" and HAVE_XGB:
        return XGBRegressor(**params)
    else:
        raise ValueError(f"Unknown or unavailable model kind: {kind}")


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    w = np.ones_like(y_true, dtype=float)
    w[y_true >= 5.0] = 1.5
    w[(y_true >= 3.0) & (y_true < 5.0)] = 1.2
    return np.average(np.abs(y_true - y_pred), weights=w)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    models_dir = Path(args.models)
    ensure_dir(models_dir)

    positions = [p.strip().upper() for p in args.positions.split(",")]
    meta_cols = [c.strip() for c in args.meta_cols.split(",")]
    target_col = args.target_col

    df = pd.read_csv(args.train)
    needed = set(meta_cols + [target_col])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV missing required columns: {missing}")

    # global feature space (union across positions)
    all_feature_cols = select_feature_columns(df, meta_cols, target_col)

    X_all = df[all_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    xscaler = MinMaxScaler()
    X_all_scaled = xscaler.fit_transform(X_all)
    if not hasattr(xscaler, "feature_names_in_"):
        xscaler.feature_names_in_ = np.array(all_feature_cols)

    y_all = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    yscaler = StandardScaler()
    y_all_scaled = yscaler.fit_transform(y_all).reshape(-1)

    joblib.dump(xscaler, models_dir / "xscaler.save")
    joblib.dump(yscaler, models_dir / "yscaler.save")

    features_map: Dict[str, List[str]] = {}
    grid = candidate_grid()

    for pos in positions:
        pdf = df[df["position"].str.upper() == pos].copy()
        if pdf.empty:
            print(f"[train] WARNING: no rows for position {pos}; skipping", flush=True)
            continue

        feats = [c for c in all_feature_cols if c in pdf.columns]
        features_map[pos] = feats

        X_pos_all = pdf[all_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        X_pos_scaled = xscaler.transform(X_pos_all).astype(np.float32)
        feats_idx = [all_feature_cols.index(f) for f in feats]
        X_pos = X_pos_scaled[:, feats_idx]

        y_pos = pdf[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        y_pos_scaled = yscaler.transform(y_pos).reshape(-1)

        groups = pdf["team"].astype(str).to_numpy()
        gkf = GroupKFold(n_splits=args.cv)

        fold = 0
        for train_idx, val_idx in gkf.split(X_pos, y_pos_scaled, groups=groups):
            fold += 1
            cv_dir = models_dir / f"cv{fold}_{pos}"
            ensure_dir(cv_dir)

            results = []
            cand_id = 0
            for kind, params in grid:
                cand_id += 1
                model = make_model(kind, params)
                try:
                    model.fit(X_pos[train_idx], y_pos_scaled[train_idx])
                    pred_scaled = model.predict(X_pos[val_idx])
                    mae = mean_absolute_error(y_pos_scaled[val_idx], pred_scaled)
                    pred = y_pos[val_idx].copy()
                    pred_unscaled = yscaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
                    wt = weighted_mae(pdf.iloc[val_idx][target_col].to_numpy(), pred_unscaled)
                except Exception as e:
                    print(f"[train] Fold {fold} {pos} cand {cand_id} FAILED: {e}", flush=True)
                    continue
                results.append({
                    "cand_id": cand_id,
                    "kind": kind,
                    "params": params,
                    "mae_scaled": float(mae),
                    "wmae_unscaled": float(wt)
                })

            if not results:
                print(f"[train] Fold {fold} {pos}: no successful candidates", flush=True)
                continue

            results_sorted = sorted(results, key=lambda r: r["wmae_unscaled"])
            keep = results_sorted[:args.top_k]

            summary_lines = []
            for i, rec in enumerate(keep, start=1):
                cid = str(i).zfill(4)
                rec_dir = cv_dir / cid
                ensure_dir(rec_dir)
                model = make_model(rec["kind"], rec["params"])
                model.fit(X_pos[train_idx], y_pos_scaled[train_idx])
                joblib.dump(model, rec_dir / "model.pkl")
                with open(rec_dir / "meta.json", "w") as f:
                    json.dump(rec, f, indent=2)
                summary_lines.append(f"Candidate {cid} {rec['kind']} WMAE={rec['wmae_unscaled']:.4f} params={rec['params']}")

            with open(cv_dir / "search.txt", "w") as f:
                f.write("Top candidates (by weighted MAE in unscaled space)\n")
                f.write("\n".join(summary_lines) + "\n")

            print(f"[train] Fold {fold} {pos}: saved {len(keep)} candidates", flush=True)

    joblib.dump(features_map, models_dir / "features.save")
    print(f"[train] Artifacts saved in: {models_dir}", flush=True)


if __name__ == "__main__":
    main()
