
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def _season_rank(value):
    if pd.isna(value):
        return -1
    text = str(value).strip()
    if not text:
        return -1
    head = text.split('-')[0]
    if head.isdigit():
        return int(head)
    digits = ''.join(ch for ch in text if ch.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    return -1


# File paths
input_path = os.path.join('data', 'cleaned_merged_seasons.csv')
samples_path = os.path.join('data', 'samples.csv')

# Read input data
raw = pd.read_csv(input_path)

# Detect latest season/GW present in the cleaned data
raw['_season_rank'] = raw['season_x'].apply(_season_rank)
raw['_gw_numeric'] = pd.to_numeric(raw['GW'], errors='coerce').fillna(0).astype(int)
latest_record = raw.sort_values(['_season_rank', '_gw_numeric']).iloc[-1]
latest_season = str(latest_record['season_x'])
latest_gw = int(latest_record['_gw_numeric'])
output_path = os.path.join('data', f'samples_{latest_season}GW{latest_gw}.csv')

# Remove helper columns before further processing
raw = raw.drop(columns=['_season_rank', '_gw_numeric'])

# Columns for output (from samples.csv)
sample_header = pd.read_csv(samples_path, nrows=0).columns.tolist()

# Define rolling windows
windows = [1, 3, 5, 10, 38]

# Map columns from cleaned_merged_seasons to samples.csv
col_map = {
    'season': 'season_x',
    'gw': 'GW',
    'position': 'position',
    'player': 'name',
    'team': 'team_x',
    'opponent': 'opp_team_name',
    'home': 'was_home',
    'player fpl points': 'total_points',
    'player minutes played': 'minutes',
    'player influence': 'influence',
    'player creativity': 'creativity',
    'player threat': 'threat',
    'player goals scored': 'goals_scored',
    'player penalties missed': 'penalties_missed',
    'player assists': 'assists',
    'player goals conceded': 'goals_conceded',
    'player own goals': 'own_goals',
    'player saves': 'saves',
    'player penalties saved': 'penalties_saved',
    'player yellow cards': 'yellow_cards',
    'player red cards': 'red_cards',
    'player bps': 'bps',
    'player fpl bonus points': 'bonus',
    # Add more mappings as needed
}

# Group by player/season for rolling
raw['season'] = raw['season_x']
raw['gw'] = raw['GW']
raw['player'] = raw['name']
raw['team'] = raw['team_x']
raw['opponent'] = raw['opp_team_name']
raw['home'] = raw['was_home']

# Sort for rolling
raw = raw.sort_values(['season', 'player', 'gw'])

# For each player/season, compute rolling stats
groups = raw.groupby(['season', 'player'])

rows = []
total_players = len(groups)
print(f"Processing {total_players} player/season groups...")

for (season, player), df in tqdm(groups, desc="Players", unit="player"):
    df = df.sort_values('gw')
    for idx, row in df.iterrows():
        out = {}
        out['season'] = season
        out['gw'] = row['gw']
        out['position'] = row['position']
        out['player'] = player
        out['team'] = row['team']
        out['opponent'] = row['opponent']
        out['home'] = row['home']
        # Rolling stats (NO LEAKAGE: use only GWs < current)
        mask = df['gw'] < row['gw']
        df_hist = df[mask]
        for stat, src in [
            ('player fpl points', 'total_points'),
            ('player relevant fpl points', 'total_points'),
            ('player minutes played', 'minutes'),
            ('player influence', 'influence'),
            ('player creativity', 'creativity'),
            ('player threat', 'threat'),
            ('player goals scored', 'goals_scored'),
            ('player penalties missed', 'penalties_missed'),
            ('player assists', 'assists'),
            ('player goals conceded', 'goals_conceded'),
            ('player own goals', 'own_goals'),
            ('player saves', 'saves'),
            ('player penalties saved', 'penalties_saved'),
            ('player yellow cards', 'yellow_cards'),
            ('player red cards', 'red_cards'),
            ('player bps', 'bps'),
            ('player fpl bonus points', 'bonus'),
        ]:
            vals = [df_hist[src].tail(w).sum() if not df_hist.empty else 0 for w in windows]
            for i, w in enumerate(windows):
                out[f'{stat} {w}'] = vals[i]
        # Fill missing columns with 0 or ''
        for col in sample_header:
            if col not in out:
                if 'player' in col or 'team' in col or 'opponent' in col or col in ['season', 'gw', 'position', 'player', 'team', 'opponent', 'home']:
                    out[col] = ''
                else:
                    out[col] = 0
        rows.append(out)

# Write output
out_df = pd.DataFrame(rows)
out_df = out_df[sample_header]
out_df.to_csv(output_path, index=False)

print(f'Wrote output to {output_path}')
