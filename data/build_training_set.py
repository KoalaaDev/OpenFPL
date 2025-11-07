#!/usr/bin/env python3
"""
Build a feature-ready training CSV from cleaned_merged_seasons.csv with no leakage.

Output schema (example):
  season, gw, position, player, team, opponent, home,
  player fpl points {1,3,5,10,38}, ... (other player rolling metrics),
  team goals scored {1,3,5,10,38}, team goals conceded {1,3,5,10,38},
  opponent goals scored {1,3,5,10,38}, opponent goals conceded {1,3,5,10,38},
  target

Notes:
  - All rolling features are computed with shift(1) to exclude the current GW (no leakage)
  - Team goals are derived from team_h_score/team_a_score depending on home/away
  - Opponent aggregates come from the opponent team's rolling values
  - This produces a good minimal set of features compatible with train_openfpl.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


PLAYER_METRICS = [
    ("player fpl points", "total_points"),
    ("player relevant fpl points", "total_points"),  # proxy
    ("player minutes played", "minutes"),
    ("player influence", "influence"),
    ("player creativity", "creativity"),
    ("player threat", "threat"),
    ("player goals scored", "goals_scored"),
    ("player penalties missed", "penalties_missed"),
    ("player assists", "assists"),
    ("player goals conceded", "goals_conceded"),
    ("player own goals", "own_goals"),
    ("player saves", "saves"),
    ("player penalties saved", "penalties_saved"),
    ("player yellow cards", "yellow_cards"),
    ("player red cards", "red_cards"),
    ("player bps", "bps"),
    ("player fpl bonus points", "bonus"),
]


WINDOWS = [1, 3, 5, 10, 38]


def parse_args():
    ap = argparse.ArgumentParser(description="Build a feature-ready training CSV up to last GW")
    ap.add_argument("--cleaned", default="data/cleaned_merged_seasons.csv", help="Path to cleaned merged seasons CSV")
    ap.add_argument("--out", default="data/train_up_to_last_gw.csv", help="Output training CSV path")
    ap.add_argument("--season", help="Optional season to filter (e.g., 2025-26). If not provided, uses all seasons present.")
    return ap.parse_args()


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    args = parse_args()
    df = pd.read_csv(args.cleaned, low_memory=False)

    # Normalize columns
    season_col = "season" if "season" in df.columns else "season_x"
    name_col = "name"
    pos_col = "position"
    team_col = "team" if "team" in df.columns else "team_x"
    opp_col = "opp_team_name"
    home_col = "was_home"
    gw_col = "GW"

    # Rename into canonical columns used here
    df = df.rename(columns={season_col: "season", team_col: "team", opp_col: "opponent", home_col: "home"})
    df = df.rename(columns={gw_col: "gw", name_col: "player", pos_col: "position"})

    # Optional filter by season
    if args.season:
        df = df[df["season"].astype(str) == args.season]
        if df.empty:
            raise ValueError(f"No rows found for season={args.season}")

    # Ensure numeric fields
    for _, src in PLAYER_METRICS:
        if src in df.columns:
            df[src] = safe_num(df[src]).fillna(0.0)
    # Team score columns
    for col in ["team_h_score", "team_a_score"]:
        if col in df.columns:
            df[col] = safe_num(df[col]).fillna(0.0)
        else:
            df[col] = 0.0

    # Derive per-row team goals for/against for this match
    df["home"] = df["home"].astype(bool)
    df["team_goals_for"] = np.where(df["home"], df["team_h_score"], df["team_a_score"]).astype(float)
    df["team_goals_against"] = np.where(df["home"], df["team_a_score"], df["team_h_score"]).astype(float)

    # Unique team-game rows (avoid duplicate per player)
    team_game = df[["season","gw","team","team_goals_for","team_goals_against"]].drop_duplicates(["season","gw","team"]).copy()
    team_game = team_game.sort_values(["season","team","gw"]).reset_index(drop=True)

    # Compute rolling team aggregates with no leakage (shift 1)
    team_roll = []
    for (season, team), g in team_game.groupby(["season","team"], sort=False):
        g = g.sort_values("gw").copy()
        out = g[["season","team","gw"]].copy()
        for w in WINDOWS:
            out[f"team goals scored {w}"] = g["team_goals_for"].rolling(window=w, min_periods=1).sum().shift(1).fillna(0.0).values
            out[f"team goals conceded {w}"] = g["team_goals_against"].rolling(window=w, min_periods=1).sum().shift(1).fillna(0.0).values
        team_roll.append(out)
    team_roll = pd.concat(team_roll, ignore_index=True) if team_roll else pd.DataFrame()

    # Opponent rolls: reuse team_roll by joining on opponent as team
    opp_roll = team_roll.copy()
    # Prefix for opponent columns
    opp_cols = {c: c.replace("team ", "opponent ") for c in opp_roll.columns if c.startswith("team ")}
    opp_roll = opp_roll.rename(columns={**{"team": "opponent"}, **opp_cols})

    # Player-level rolling features
    df = df.sort_values(["season","player","gw"]).reset_index(drop=True)
    player_rows = []
    for (season, player), g in df.groupby(["season","player"], sort=False):
        g = g.sort_values("gw").copy()
        out = g[["season","gw","position","player","team","opponent","home"]].copy()
        for label, src in PLAYER_METRICS:
            if src not in g.columns:
                vals = pd.Series([0.0]*len(g))
            else:
                vals = pd.to_numeric(g[src], errors="coerce").fillna(0.0)
            for w in WINDOWS:
                out[f"{label} {w}"] = vals.rolling(window=w, min_periods=1).sum().shift(1).fillna(0.0).values
        # target is current match points
        out["target"] = pd.to_numeric(g["total_points"], errors="coerce").fillna(0.0).values
        player_rows.append(out)
    players_roll = pd.concat(player_rows, ignore_index=True)

    # Merge team and opponent aggregates to player rows on (season,gw,team/opponent)
    train = players_roll.merge(team_roll, how="left", on=["season","gw","team"]).merge(
        opp_roll, how="left", on=["season","gw","opponent"]) 
    # Fill NaNs for aggregates with 0
    agg_cols = [c for c in train.columns if any(c.startswith(p) for p in ["team ", "opponent "])]
    train[agg_cols] = train[agg_cols].fillna(0.0)

    # Remove GW==1 rows (no prior info) if desired; keep them with zeros is also fine
    # Keep all rows.

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_path, index=False)
    # Report last season/gw
    last_season = str(sorted(train['season'].unique())[-1])
    last_gw = int(pd.to_numeric(train[train['season']==last_season]['gw'], errors='coerce').max())
    print(f"Wrote {out_path} ({len(train)} rows). Latest season {last_season} up to GW {last_gw}.")


if __name__ == "__main__":
    main()
