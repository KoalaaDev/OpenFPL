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

Usage:

Get the training set first, e.g.,
# All seasons included (good default)
python data\build_training_set.py --out data\train_up_to_last_gw.csv

# Or restrict to just the current season if you prefer
python data\build_training_set.py --season 2025-26 --out data\train_2025-26_up_to_last_gw.csv

Training:
# Train on the new dataset (all seasons)
python train_openfpl.py --train data\train_up_to_last_gw.csv --models models_lastGW --positions GK,DEF,MID,FWD --target-col target --cv 5 --top-k 10

# Or if you built a season-specific file
python train_openfpl.py --train data\train_2025-26_up_to_last_gw.csv --models models_2025-26_lastGW --positions GK,DEF,MID,FWD --target-col target --cv 5 --top-k 10

After training, run:
  python predict_openfpl.py --samples data/samples_gwN.csv --models models_custom --out data/preds_gwN.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from tqdm.auto import tqdm

try:
    import cupy as cp
    HAVE_CUPY = True
except Exception:
    cp = None
    HAVE_CUPY = False

try:
    from cuml.ensemble import RandomForestRegressor as cuMLRandomForestRegressor
    HAVE_CUML_RF = True
except Exception:
    cuMLRandomForestRegressor = None
    HAVE_CUML_RF = False

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings

# Optional XGBoost
try:
    import xgboost as xgb
    from xgboost import XGBRegressor, core as xgb_core
    HAVE_XGB = True
    try:
        _ver_parts = xgb.__version__.split("+")[0].split(".")
        XGB_VERSION = tuple(int(part) for part in _ver_parts[:3])
    except Exception:
        XGB_VERSION = (0, 0, 0)
except Exception:
    xgb = None
    XGBRegressor = None
    xgb_core = None
    HAVE_XGB = False
    XGB_VERSION = (0, 0, 0)


def _detect_xgb_gpu() -> bool:
    if not HAVE_XGB:
        return False
    if not HAVE_CUPY:
        warnings.warn("CuPy not available; falling back to CPU for XGBoost.")
        return False
    try:
        if hasattr(xgb_core, "_has_cuda_support"):
            if not xgb_core._has_cuda_support():
                return False
        # Ensure a visible device exists (CUDA_VISIBLE_DEVICES=-1 disables)
        cuda_mask = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_mask is not None and cuda_mask.strip() in {"", "-1"}:
            return False
        # Probe a tiny fit to confirm runtime availability
        probe_params = {"n_estimators": 1, "max_depth": 1, "verbosity": 0}
        if XGB_VERSION >= (2, 0, 0):
            probe_params.update({"device": "cuda", "tree_method": "hist"})
        else:
            probe_params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
        dummy = XGBRegressor(**probe_params)
        dummy.fit(cp.zeros((1, 1), dtype=cp.float32), cp.zeros(1, dtype=cp.float32))
        return True
    except Exception:
        warnings.warn("Falling back to CPU for XGBoost (GPU unavailable).")
        return False


USE_XGB_GPU = _detect_xgb_gpu()


def _detect_cuml_rf_gpu() -> bool:
    if not HAVE_CUML_RF or not HAVE_CUPY:
        return False
    try:
        probe = cuMLRandomForestRegressor(n_estimators=1, max_depth=2, random_state=0)
        X_dummy = cp.zeros((4, 2), dtype=cp.float32)
        y_dummy = cp.zeros(4, dtype=cp.float32)
        probe.fit(X_dummy, y_dummy)
        return True
    except Exception:
        warnings.warn("Falling back to CPU RandomForest (cuML GPU unavailable).")
        return False


USE_GPU_RF = _detect_cuml_rf_gpu()


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
    rf_param_sets: List[dict] = []
    for n in [300, 600]:
        for depth in [None, 10, 16]:
            for leaf in [1, 3]:
                rf_param_sets.append({
                    "n_estimators": n,
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "random_state": 0,
                })
    if USE_GPU_RF:
        for params in rf_param_sets:
            grid.append(("rf_gpu", params.copy()))
    else:
        for params in rf_param_sets:
            cpu_params = params.copy()
            cpu_params["n_jobs"] = -1
            grid.append(("rf", cpu_params))
    if HAVE_XGB:
        for n in [400, 800]:
            for depth in [4, 6, 8]:
                for lr in [0.05, 0.1]:
                    params = {
                        "n_estimators": n,
                        "max_depth": depth,
                        "learning_rate": lr,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "reg_lambda": 1.0,
                        "random_state": 0,
                    }
                    if USE_XGB_GPU:
                        if XGB_VERSION >= (2, 0, 0):
                            params.update({"device": "cuda", "tree_method": "hist"})
                        else:
                            params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
                    else:
                        params.update({"tree_method": "hist", "n_jobs": -1})
                    grid.append(("xgb", params))
    return grid


def make_model(kind: str, params: dict):
    if kind == "rf":
        return RandomForestRegressor(**params)
    elif kind == "rf_gpu" and HAVE_CUML_RF:
        return cuMLRandomForestRegressor(**params)
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

    if HAVE_XGB:
        backend = "GPU" if USE_XGB_GPU else "CPU"
        print(f"[train] XGBoost backend: {backend}", flush=True)
    if USE_GPU_RF:
        print("[train] RandomForest candidates will use cuML on GPU.", flush=True)
    elif HAVE_CUML_RF and not USE_GPU_RF:
        print("[train] cuML RandomForest detected but GPU execution unavailable; staying on CPU.", flush=True)
    positions = [p.strip().upper() for p in args.positions.split(",")]
    meta_cols = [c.strip() for c in args.meta_cols.split(",")]
    target_col = args.target_col

    df = pd.read_csv(args.train, low_memory=False)
    needed = set(meta_cols + [target_col])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV missing required columns: {missing}")

    # global feature space (union across positions)
    all_feature_cols = select_feature_columns(df, meta_cols, target_col)

    X_all_df = df[all_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xscaler = MinMaxScaler()
    X_all_scaled = xscaler.fit_transform(X_all_df).astype(np.float32)
    if not hasattr(xscaler, "feature_names_in_"):
        xscaler.feature_names_in_ = np.array(all_feature_cols)

    y_all = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    yscaler = StandardScaler()
    y_all_scaled = yscaler.fit_transform(y_all).reshape(-1)

    joblib.dump(xscaler, models_dir / "xscaler.save")
    joblib.dump(yscaler, models_dir / "yscaler.save")

    features_map: Dict[str, List[str]] = {}
    grid = candidate_grid()

    for pos in tqdm(positions, desc="Positions", unit="pos"):
        pdf = df[df["position"].str.upper() == pos].copy()
        if pdf.empty:
            tqdm.write(f"[train] WARNING: no rows for position {pos}; skipping")
            continue

        feats = [c for c in all_feature_cols if c in pdf.columns]
        features_map[pos] = feats

        X_pos_all_df = pdf[all_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_pos_scaled = xscaler.transform(X_pos_all_df).astype(np.float32)
        feats_idx = [all_feature_cols.index(f) for f in feats]
        X_pos = X_pos_scaled[:, feats_idx]

        y_pos = pdf[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        y_pos_scaled = yscaler.transform(y_pos).reshape(-1)
        groups = pdf["team"].astype(str).to_numpy()
        gkf = GroupKFold(n_splits=args.cv)
        splits = list(gkf.split(X_pos, y_pos_scaled, groups=groups))

        fold_iter = tqdm(enumerate(splits, start=1), total=len(splits), desc=f"{pos}: folds", unit="fold", leave=False)
        for fold, (train_idx, val_idx) in fold_iter:
            cv_dir = models_dir / f"cv{fold}_{pos}"
            ensure_dir(cv_dir)

            results = []
            cand_iter = tqdm(enumerate(grid, start=1), total=len(grid), desc=f"{pos} F{fold}: candidates", unit="model", leave=False)
            for cand_id, (kind, params) in cand_iter:
                model = make_model(kind, params)
                use_gpu_model = ((kind == "xgb" and USE_XGB_GPU) or (kind == "rf_gpu" and USE_GPU_RF))
                try:
                    X_train = X_pos[train_idx]
                    y_train = y_pos_scaled[train_idx]
                    X_val_input = X_pos[val_idx]

                    if use_gpu_model:
                        X_train = cp.asarray(X_train)
                        y_train = cp.asarray(y_train)
                        X_val_input = cp.asarray(X_val_input)

                    cand_iter.set_postfix_str(kind)
                    model.fit(X_train, y_train)
                    pred_scaled = model.predict(X_val_input)
                    if use_gpu_model and hasattr(pred_scaled, "__cuda_array_interface__"):
                        pred_scaled = cp.asnumpy(pred_scaled)

                    mae = mean_absolute_error(y_pos_scaled[val_idx], pred_scaled)
                    pred = y_pos[val_idx].copy()
                    pred_unscaled = yscaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
                    wt = weighted_mae(pdf.iloc[val_idx][target_col].to_numpy(), pred_unscaled)
                except Exception as e:
                    tqdm.write(f"[train] Fold {fold} {pos} cand {cand_id} FAILED: {e}")
                    continue
                results.append({
                    "cand_id": cand_id,
                    "kind": kind,
                    "params": params,
                    "mae_scaled": float(mae),
                    "wmae_unscaled": float(wt)
                })
            cand_iter.close()

            if not results:
                tqdm.write(f"[train] Fold {fold} {pos}: no successful candidates")
                continue

            results_sorted = sorted(results, key=lambda r: r["wmae_unscaled"])
            keep = results_sorted[:args.top_k]

            summary_lines = []
            for i, rec in enumerate(keep, start=1):
                cid = str(i).zfill(4)
                rec_dir = cv_dir / cid
                ensure_dir(rec_dir)
                model = make_model(rec["kind"], rec["params"])
                use_gpu_model = ((rec["kind"] == "xgb" and USE_XGB_GPU) or (rec["kind"] == "rf_gpu" and USE_GPU_RF))
                X_train = X_pos[train_idx]
                y_train = y_pos_scaled[train_idx]
                if use_gpu_model:
                    X_train = cp.asarray(X_train)
                    y_train = cp.asarray(y_train)
                model.fit(X_train, y_train)
                joblib.dump(model, rec_dir / "model.pkl")
                with open(rec_dir / "meta.json", "w") as f:
                    json.dump(rec, f, indent=2)
                summary_lines.append(f"Candidate {cid} {rec['kind']} WMAE={rec['wmae_unscaled']:.4f} params={rec['params']}")

            with open(cv_dir / "search.txt", "w") as f:
                f.write("Top candidates (by weighted MAE in unscaled space)\n")
                f.write("\n".join(summary_lines) + "\n")

            tqdm.write(f"[train] Fold {fold} {pos}: saved {len(keep)} candidates")
        fold_iter.close()

    joblib.dump(features_map, models_dir / "features.save")
    tqdm.write(f"[train] Artifacts saved in: {models_dir}")


if __name__ == "__main__":
    main()
