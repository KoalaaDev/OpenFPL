import joblib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
features = joblib.load(root / 'models' / 'features.save')

for pos, cols in features.items():
    fams = {'player':0,'team':0,'opponent':0,'status':0,'other':0}
    for c in cols:
        lc = c.lower()
        if lc.startswith('player '): fams['player'] += 1
        elif lc.startswith('team '): fams['team'] += 1
        elif lc.startswith('opponent '): fams['opponent'] += 1
        elif lc.startswith('status '): fams['status'] += 1
        else: fams['other'] += 1
    total = len(cols)
    print(f"{pos}: total {total} | player {fams['player']} | team {fams['team']} | opponent {fams['opponent']} | status {fams['status']} | other {fams['other']}")
