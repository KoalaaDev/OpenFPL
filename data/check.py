import joblib
import pandas as pd
from pathlib import Path
import os

# Adjust season/GW for the file you want to check
SAMPLES_FILE = 'samples_2025-26GW5.csv'

here = Path(__file__).resolve().parent
repo_root = here.parent
models_dir = repo_root / 'models'
samples_path = here / SAMPLES_FILE

xscaler = joblib.load(models_dir / 'xscaler.save')
features = joblib.load(models_dir / 'features.save')

samples = pd.read_csv(samples_path)
if hasattr(xscaler, 'feature_names_in_'):
    xfeats = list(xscaler.feature_names_in_)
else:
    xfeats = []

missing = [f for f in xfeats if f not in samples.columns]
print('Missing features from xscaler:', len(missing))
if missing:
    print('First 25 missing:', missing[:25])

for pos, flist in features.items():
    present = [f for f in flist if f in samples.columns]
    nunique_sum = int(samples[present].nunique().sum()) if present else 0
    print(f"{pos} present {len(present)}/{len(flist)}, sum nunique {nunique_sum}")

# Show a small slice of key player features to confirm they vary
for pos in ['GK','DEF','MID','FWD']:
    dfp = samples[samples['position'].str.upper()==pos]
    if dfp.empty:
        continue
    cols = [c for c in dfp.columns if c.startswith('player fpl points')][:3]
    cols += [c for c in dfp.columns if c.startswith('player minutes played')][:3]
    cols = ['player','team','opponent'] + cols
    print(f"\n{pos} sample rows:\n", dfp[cols].head(10).to_string(index=False))