#!/usr/bin/env python3
"""
OpenFPL — Predict upcoming GW points from pretrained artifacts.

This is the script version of your notebook (`play.ipynb`). It:
  1) Loads pretrained models (per position, across CV folds & candidates)
  2) Loads x/y scalers and the position-specific feature lists
  3) Reads a samples CSV (one row per *entity* for the target GW)
  4) Preprocesses per position (scale & select features)
  5) Predicts with every model and MEDIANS the ensemble
  6) Writes a predictions CSV (season, gw, position, player, team, opponent, home, prediction)

Typical usage:
    python predict_openfpl.py \
        --samples data/samples.csv \
        --models  models \
        --out     data/predictions.csv \
        --exclude-am

Expected directory layout (same as your notebook):
    models/
      xscaler.save
      yscaler.save
      features.save              # dict[str position -> list[str feature names]]
      cv1_GK/ search.txt  0001/<model.pkl>  0002/<model.pkl> ...
      cv1_DEF/ ...
      ...
      cv5_FWD/ ...

The samples CSV must include:
    season, gw, position, player, team, opponent, home, <feature columns>

Notes:
- "AM" stands for the manager/assistant-manager entity used as a *feature* in the paper; it is not a pickable FPL player.
  Use --exclude-am to remove these rows from the output.
- Standard run command: python predict_openfpl.py --samples data/samples_2025-26GW4.csv --models models --out data/predictions_2025-26GW4.csv --exclude-am
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def _debug(msg: str) -> None:
    print(f"[openfpl] {msg}", file=sys.stderr)


def load_artifacts(models_dir: Path, positions: List[str], num_cvs: int) -> Tuple[Dict[int, Dict[str, list]], object, list, object, Dict[str, List[str]]]:
    """
    Returns:
        models: dict[cv -> dict[position -> list[model]]]
        xscaler: fitted scaler with .transform and .feature_names_in_
        xscaler_features: list[str] in the order the scaler expects
        yscaler: scaler to inverse_transform target
        features: dict[position -> list[str feature names]] (subset of xscaler_features)
    """
    # Load scalers and features
    xscaler = joblib.load(models_dir / "xscaler.save")
    # Be robust if feature_names_in_ is not present
    if hasattr(xscaler, "feature_names_in_"):
        xscaler_features = list(xscaler.feature_names_in_)
    else:
        # Fallback: try to read from a sibling file cached by training code
        feats_path = models_dir / "xscaler_feature_names.json"
        if feats_path.exists():
            xscaler_features = list(pd.read_json(feats_path).iloc[:,0].tolist())
        else:
            raise RuntimeError("xscaler has no .feature_names_in_. Please export feature names during training.")
    yscaler = joblib.load(models_dir / "yscaler.save")
    features = joblib.load(models_dir / "features.save")

    # Load models: replicate notebook's 'search.txt' parsing.
    models: Dict[int, Dict[str, list]] = {cv: {pos: [] for pos in positions} for cv in range(1, num_cvs+1)}

    for cv in range(1, num_cvs + 1):
        for pos in positions:
            cv_pos_dir = models_dir / f"cv{cv}_{pos}"
            if not cv_pos_dir.exists():
                _debug(f"WARNING: missing {cv_pos_dir}, skipping")
                continue

            loaded_any = False
            search_txt = cv_pos_dir / "search.txt"
            candidate_ids: List[str] = []

            if search_txt.exists():
                try:
                    text = search_txt.read_text()
                    # The notebook builds candidates from the tail lines after "The population is:"
                    # then splits on "Candidate NNN". We'll mirror that.
                    tail = text.split("The population is:")[-1]
                    parts = tail.split("Candidate ")[1:]  # each starts with "<num> ..."
                    for part in parts:
                        cid = part.split(" ")[0].strip()
                        if cid.isdigit():
                            candidate_ids.append(cid)
                except Exception as e:
                    _debug(f"WARNING: failed to parse {search_txt}: {e}")

            # Fallback: if parsing failed, load all subdirs
            if not candidate_ids:
                candidate_ids = [d.name for d in cv_pos_dir.iterdir() if d.is_dir() and d.name.isdigit()]

            for cid in candidate_ids:
                cdir = cv_pos_dir / cid
                if not cdir.exists() or not cdir.is_dir():
                    continue
                # load first .pkl-like file
                pkl_files = [p for p in cdir.iterdir() if p.suffix in [".pkl",".joblib",".sav",".model"]]
                if not pkl_files:
                    # also allow a single file directly inside
                    pkl_files = [p for p in cdir.iterdir() if p.is_file()]
                if not pkl_files:
                    continue
                try:
                    model = joblib.load(pkl_files[0])
                    models[cv][pos].append(model)
                    loaded_any = True
                except Exception as e:
                    _debug(f"WARNING: failed to load model from {pkl_files[0]}: {e}")
            if not loaded_any:
                _debug(f"WARNING: no models loaded for cv{cv} {pos}")

    return models, xscaler, xscaler_features, yscaler, features


def validate_samples_columns(df: pd.DataFrame, needed: List[str]) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Samples CSV is missing required columns: {missing}")


def preprocess_for_position(df: pd.DataFrame, xscaler, xscaler_features: List[str], features_for_pos: List[str]) -> np.ndarray:
    # Ensure features exist
    missing = [f for f in features_for_pos if f not in xscaler_features]
    if missing:
        raise ValueError(f"Features referenced by artifacts were not found in xscaler_features: {missing}")

    # Align columns in scaler order
    # Use a DataFrame with the scaler's expected column names to avoid feature-name warnings
    X_df = df[xscaler_features].copy()
    # Ensure numeric dtype and no NaNs before scaling
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df = X_df.fillna(0.0)
    X_scaled = xscaler.transform(X_df)
    X_scaled = np.nan_to_num(X_scaled).astype("float32")

    # Subselect to the features used by the trained models for this position
    indices = [xscaler_features.index(f) for f in features_for_pos]
    X_pos = X_scaled[:, indices]
    return X_pos


def predict_position(models_for_pos: Dict[int, list], X_pos: np.ndarray, yscaler) -> np.ndarray:
    all_preds = []
    for cv, model_list in models_for_pos.items():
        for model in model_list:
            try:
                yhat = model.predict(X_pos)
                # inverse transform expects 2D
                yhat = yscaler.inverse_transform(yhat.reshape(-1, 1)).reshape(-1)
                all_preds.append(yhat)
            except Exception as e:
                _debug(f"WARNING: model failed during prediction: {e}")
    if not all_preds:
        return np.zeros(X_pos.shape[0], dtype=float)
    # Ensemble by median across models
    return np.median(np.vstack(all_preds), axis=0)


def run(args: argparse.Namespace) -> int:
    positions = [p.strip().upper() for p in args.positions.split(",")]
    models, xscaler, xscaler_features, yscaler, features = load_artifacts(Path(args.models), positions, args.num_cvs)

    # Load samples
    samples = pd.read_csv(args.samples)
    base_cols = ["season","gw","position","player","team","opponent","home"]
    validate_samples_columns(samples, base_cols)

    # Prepare output
    out_rows = []

    for pos in positions:
        pos_df = samples[samples["position"].str.upper() == pos].copy()
        if pos_df.empty:
            _debug(f"NOTE: no rows for position {pos} in samples")
            continue

        feats_for_pos = features.get(pos)
        if feats_for_pos is None:
            raise KeyError(f"No feature list found for position '{pos}' in features.save")
        # Preprocess
        X_pos = preprocess_for_position(pos_df, xscaler, xscaler_features, feats_for_pos)
        # Aggregate all models for this position across CVs
        models_for_pos = {cv: models[cv][pos] for cv in models if pos in models[cv]}
        preds = predict_position(models_for_pos, X_pos, yscaler)

        pos_df = pos_df[base_cols].copy()
        pos_df["prediction"] = preds
        out_rows.append(pos_df)

    if not out_rows:
        _debug("No predictions produced (check inputs).")
        return 2

    out_df = pd.concat(out_rows, ignore_index=True)

    if args.exclude_am:
        out_df = out_df[out_df["position"] != "AM"]

    # Sort nicely
    out_df = out_df.sort_values(["gw","position","prediction"], ascending=[True, True, False])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}  ({len(out_df)} rows)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="OpenFPL pretrained prediction script (from notebook).")
    ap.add_argument("--samples", required=True, help="Path to samples CSV (one row per entity for the target GW)")
    ap.add_argument("--models",  required=True, help="Path to models artifacts directory (with cv*_*/ search.txt, xscaler.save, yscaler.save, features.save)")
    ap.add_argument("--out",     required=True, help="Where to write predictions CSV")
    ap.add_argument("--positions", default="GK,DEF,MID,FWD,AM", help="Comma-separated list of positions to score")
    ap.add_argument("--num-cvs", type=int, default=5, help="Number of CV folds expected in the models directory")
    ap.add_argument("--exclude-am", action="store_true", help='Drop the "AM" manager rows from output')
    args = ap.parse_args()
    try:
        sys.exit(run(args))
    except Exception as e:
        _debug(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
