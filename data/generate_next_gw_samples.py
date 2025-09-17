import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

# Detect latest samples file
def find_latest_samples_file(data_dir):
    pattern = re.compile(r"samples_(\d{4}-\d{2})GW(\d+)\.csv")
    latest = None
    latest_season = None
    latest_gw = -1
    for fname in os.listdir(data_dir):
        m = pattern.match(fname)
        if m:
            season, gw = m.group(1), int(m.group(2))
            if (latest is None) or (season > latest_season) or (season == latest_season and gw > latest_gw):
                latest = fname
                latest_season = season
                latest_gw = gw
    return latest, latest_season, latest_gw

# Paths
DATA_DIR = 'data'
fixtures_path = os.path.join(DATA_DIR, 'fixtures_timetable.csv')

# Find latest samples file
latest_samples, season, last_gw = find_latest_samples_file(DATA_DIR)
if not latest_samples:
    raise FileNotFoundError('No samples_(season)GW(gw).csv file found in data/')

samples_path = os.path.join(DATA_DIR, latest_samples)
samples = pd.read_csv(samples_path)
fixtures = pd.read_csv(fixtures_path)

# Next GW
next_gw = last_gw + 1

# Filter fixtures for next GW and current season
fixtures_next = fixtures[(fixtures['season'] == season) & (fixtures['event'] == next_gw)]

# Prepare output rows
rows = []

# For each fixture, get home and away teams
for _, fix in fixtures_next.iterrows():
    for home_away, team, opp, is_home in [
        ('home', fix['team_h_name'], fix['team_a_name'], True),
        ('away', fix['team_a_name'], fix['team_h_name'], False)
    ]:
        # Get all players from this team in the latest samples file
        team_players = samples[(samples['team'] == team) & (samples['gw'] == last_gw)]
        for _, player_row in team_players.iterrows():
            new_row = player_row.copy()
            new_row['gw'] = next_gw
            new_row['opponent'] = opp
            new_row['home'] = is_home
            # Set all rolling/stat features to 0 or blank (optional: keep last values)
            for col in samples.columns:
                if re.search(r'\d$', col) and ('player' in col or 'team' in col or 'opponent' in col):
                    new_row[col] = ''
                elif re.search(r'\d$', col):
                    new_row[col] = 0
            rows.append(new_row)

# Output path
out_path = os.path.join(DATA_DIR, f'samples_{season}GW{next_gw}.csv')
out_df = pd.DataFrame(rows)
out_df.to_csv(out_path, index=False)
print(f'Wrote {out_path} ({len(out_df)} rows)')
