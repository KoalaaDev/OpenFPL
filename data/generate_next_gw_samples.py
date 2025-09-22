import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

# Detect latest samples file
def find_samples_file_for(data_dir: str, season: str, gw: int) -> str:
    fname = f"samples_{season}GW{gw}.csv"
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected base samples file not found: {path}")
    return fname


def compute_player_rolling(clean_df: pd.DataFrame, player_name: str, last_gw: int) -> dict:
    """
    Compute next-GW player rolling features using all matches up to and including last_gw.
    Returns a dict of column -> value for columns named like 'player <metric> <window>'.
    Missing metrics fall back to zeros.
    """
    windows = [1, 3, 5, 10, 38]
    # Map samples metric labels -> cleaned column names
    metric_map = {
        'fpl points': 'total_points',
        'minutes played': 'minutes',
        'influence': 'influence',
        'creativity': 'creativity',
        'threat': 'threat',
        'goals scored': 'goals_scored',
        'penalties missed': 'penalties_missed',
        'assists': 'assists',
        'goals conceded': 'goals_conceded',
        'own goals': 'own_goals',
        'saves': 'saves',
        'penalties saved': 'penalties_saved',
        'yellow cards': 'yellow_cards',
        'red cards': 'red_cards',
        'bps': 'bps',
        'fpl bonus points': 'bonus',
    }

    # Filter player's history for this season up to last_gw inclusive
    pdf = clean_df[(clean_df['name'] == player_name) & (clean_df['GW'] <= last_gw)].sort_values('GW')
    out = {}
    for label, src_col in metric_map.items():
        if src_col not in pdf.columns:
            # If column absent in cleaned data, leave zeros
            series = pd.Series(dtype=float)
        else:
            series = pd.to_numeric(pdf[src_col], errors='coerce').fillna(0.0)
        for w in windows:
            val = float(series.tail(w).sum()) if not series.empty else 0.0
            out[f'player {label} {w}'] = val
    # Best-effort proxy: relevant fpl points ~= fpl points if unknown
    for w in windows:
        out[f'player relevant fpl points {w}'] = out.get(f'player fpl points {w}', 0.0)
    return out


def main():
    # Paths
    DATA_DIR = 'data'
    fixtures_path = os.path.join(DATA_DIR, 'fixtures_timetable.csv')
    cleaned_path = os.path.join(DATA_DIR, 'cleaned_merged_seasons.csv')

    # Load fixtures and cleaned historical data
    fixtures = pd.read_csv(fixtures_path)
    cleaned = pd.read_csv(cleaned_path)

    # Normalize column names that might differ across seasons/files
    # Ensure columns 'season', 'GW', 'name', 'opp_team_name' exist
    # cleaned file has 'season_x' and 'GW'
    if 'season' in cleaned.columns:
        cleaned['season_norm'] = cleaned['season']
    else:
        cleaned['season_norm'] = cleaned.get('season_x', '')

    # Determine latest season and last played GW from cleaned data
    all_seasons = cleaned['season_norm'].dropna().unique().tolist()
    if not all_seasons:
        raise ValueError('No seasons found in cleaned_merged_seasons.csv')
    season = sorted(all_seasons)[-1]
    cleaned_season = cleaned[cleaned['season_norm'] == season].copy()
    if cleaned_season.empty:
        raise ValueError(f'No rows for season {season} in cleaned_merged_seasons.csv')
    last_gw = int(pd.to_numeric(cleaned_season['GW'], errors='coerce').max())
    next_gw = last_gw + 1

    # Read base samples for last_gw of this season to clone meta columns and non-player features
    base_samples_fname = find_samples_file_for(DATA_DIR, season, last_gw)
    samples_path = os.path.join(DATA_DIR, base_samples_fname)
    samples = pd.read_csv(samples_path)

    # Filter fixtures for next GW and current season
    fixtures_next = fixtures[(fixtures['season'] == season) & (fixtures['event'] == next_gw)]
    if fixtures_next.empty:
        raise ValueError(f'No fixtures found in fixtures_timetable.csv for season={season} event={next_gw}')

    # Prepare output rows
    rows = []

    # For each fixture, get home and away teams and clone last GW row per player, then update features
    for _, fix in fixtures_next.iterrows():
        for team, opp, is_home in [
            (fix['team_h_name'], fix['team_a_name'], True),
            (fix['team_a_name'], fix['team_h_name'], False)
        ]:
            # Get all players from this team in the latest samples file (last_gw rows)
            team_players = samples[(samples['team'] == team) & (samples['gw'] == last_gw) & (samples['season'] == season)]
            if team_players.empty:
                # If no rows (e.g., promoted teams or naming mismatches), skip gracefully
                continue
            for _, player_row in team_players.iterrows():
                new_row = player_row.copy()
                new_row['season'] = season
                new_row['gw'] = next_gw
                new_row['opponent'] = opp
                new_row['home'] = is_home

                # Compute per-player rolling stats up to last_gw (inclusive)
                pname = player_row['player']
                try:
                    rolling_vals = compute_player_rolling(cleaned_season, pname, last_gw)
                except Exception:
                    rolling_vals = {}
                # Apply rolling values to any matching columns
                for col, val in rolling_vals.items():
                    if col in new_row.index:
                        new_row[col] = val
                # For any player-* rolling columns not set, fill zeros to avoid NaNs
                for col in new_row.index:
                    if isinstance(col, str) and col.startswith('player ') and re.search(r'\s(1|3|5|10|38)$', col):
                        if pd.isna(new_row[col]):
                            new_row[col] = 0.0

                rows.append(new_row)

    # Output path
    out_path = os.path.join(DATA_DIR, f'samples_{season}GW{next_gw}.csv')
    out_df = pd.DataFrame(rows)
    # Ensure same column order as original samples for compatibility
    out_df = out_df[samples.columns]
    # Fill any remaining NaNs in player rolling columns with zeros (defensive)
    player_cols = [c for c in out_df.columns if isinstance(c, str) and c.startswith('player ') and re.search(r'\s(1|3|5|10|38)$', c)]
    if player_cols:
        out_df[player_cols] = out_df[player_cols].fillna(0.0)
    out_df.to_csv(out_path, index=False)
    print(f'Wrote {out_path} ({len(out_df)} rows)')


if __name__ == '__main__':
    main()
