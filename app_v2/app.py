#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

# Paths
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
DATA_DIR = ROOT / 'data'

app = Flask(__name__, template_folder=str(APP_DIR / 'templates'), static_folder=str(APP_DIR / 'static'))


# Utilities

def find_latest_predictions_file() -> Path | None:
    candidates: List[Tuple[str, int, Path]] = []
    if DATA_DIR.exists():
        for p in DATA_DIR.glob('predictions_*.csv'):
            m = re.match(r"predictions_(\d{4}-\d{2})GW(\d+)\.csv", p.name)
            if m:
                candidates.append((m.group(1), int(m.group(2)), p))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[-1][2]
    p = DATA_DIR / 'predictions.csv'
    return p if p.exists() else None


def fetch_bootstrap() -> dict:
    r = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/', timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_entry_info(entry_id: int) -> dict:
    r = requests.get(f'https://fantasy.premierleague.com/api/entry/{entry_id}/', timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_entry_picks(entry_id: int, event: int) -> dict:
    r = requests.get(f'https://fantasy.premierleague.com/api/entry/{entry_id}/event/{event}/picks/', timeout=15)
    r.raise_for_status()
    return r.json()


def detect_event(which: str = 'next') -> int:
    js = fetch_bootstrap()
    ev = None
    if which == 'current':
        for e in js.get('events', []):
            if e.get('is_current'):
                ev = e.get('id')
                break
    else:
        for e in js.get('events', []):
            if e.get('is_next'):
                ev = e.get('id')
                break
    if not ev:
        finished = [e.get('id') for e in js.get('events', []) if e.get('finished')] or [1]
        ev = max(finished)
    return int(ev)


def elements_index(js: dict) -> Dict[int, dict]:
    teams = {t['id']: t['name'] for t in js.get('teams', [])}
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    out: Dict[int, dict] = {}
    for e in js.get('elements', []):
        out[e['id']] = {
            'player': e.get('web_name') or e.get('second_name') or e.get('first_name'),
            'position': pos_map.get(e.get('element_type')),
            'team': teams.get(e.get('team')),
            'now_price': (e.get('now_cost') or 0) / 10.0,
        }
    return out


def fetch_live_prices() -> pd.DataFrame:
    js = fetch_bootstrap()
    idx = elements_index(js)
    rows = []
    for el in idx.values():
        if el['player'] and el['position'] and el['team']:
            rows.append({'player': el['player'], 'position': el['position'], 'team': el['team'], 'price': el['now_price']})
    return pd.DataFrame(rows)


def build_price_map(prices_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    mp: Dict[Tuple[str, str], float] = {}
    for _, r in prices_df.iterrows():
        key = (str(r['player']).strip(), str(r['team']).strip())
        price = float(r.get('price', 0) or 0)
        if price > 0:
            mp[key] = price
    return mp


def build_squad_from_entry(entry_id: int, event: int) -> tuple[pd.DataFrame, float]:
    boot = fetch_bootstrap()
    idx = elements_index(boot)
    info = fetch_entry_info(entry_id)
    picks_js = fetch_entry_picks(entry_id, event)

    # Prefer event bank from picks entry_history
    bank_tenths = (picks_js.get('entry_history', {}) or {}).get('bank')
    if bank_tenths in (None, ''):
        bank_tenths = info.get('last_deadline_bank') or info.get('bank') or 0
    bank = (bank_tenths or 0) / 10.0

    rows = []
    for p in picks_js.get('picks', []):
        el = idx.get(p.get('element'))
        if not el:
            continue
        sell_tenths = p.get('selling_price')
        sell_price = (sell_tenths / 10.0) if isinstance(sell_tenths, (int, float)) else el['now_price']
        rows.append({'player': el['player'], 'team': el['team'], 'position': el['position'], 'price': sell_price})
    return pd.DataFrame(rows), float(bank)


def suggest_single_transfers(pred: pd.DataFrame, squad: pd.DataFrame, prices: pd.DataFrame, bank: float, top: int = 10) -> pd.DataFrame:
    price_map = build_price_map(prices)

    pred = pred.copy()
    pred['position'] = pred['position'].str.upper()
    squad = squad.copy()
    squad['position'] = squad['position'].str.upper()
    squad['sell_price'] = pd.to_numeric(squad.get('price', 0), errors='coerce').fillna(0.0)

    team_counts = squad['team'].value_counts().to_dict()
    squad_set = set((p, t) for p, t in zip(squad['player'], squad['team']))

    out_rows: List[dict] = []
    for _, s in squad.iterrows():
        pos = s['position']
        team_out = s['team']
        player_out = s['player']
        sell_price = float(s['sell_price'])
        pred_out = float(pred[(pred['player'] == player_out) & (pred['team'] == team_out)]['prediction'].fillna(0).head(1).tolist() or [0.0])

        candidates = pred[pred['position'] == pos]
        for _, r in candidates.iterrows():
            key = (r['player'], r['team'])
            if key in squad_set:
                continue
            # 3-per-team cap
            new_counts = dict(team_counts)
            new_counts[team_out] = max(0, new_counts.get(team_out, 0) - 1)
            new_counts[r['team']] = new_counts.get(r['team'], 0) + 1
            if any(v > 3 for v in new_counts.values()):
                continue
            buy_price = price_map.get(key)
            if buy_price is None:
                continue
            net_cost = buy_price - sell_price
            if net_cost > bank + 1e-9:
                continue
            pred_in = float(r['prediction'])
            gain = pred_in - pred_out
            out_rows.append({
                'out_player': player_out,
                'out_team': team_out,
                'out_position': pos,
                'out_pred': pred_out,
                'sell_price': sell_price,
                'in_player': r['player'],
                'in_team': r['team'],
                'in_position': pos,
                'in_pred': pred_in,
                'buy_price': buy_price,
                'net_cost': net_cost,
                'gain': gain,
            })
    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows).sort_values(['gain', 'in_pred'], ascending=[False, False]).head(top)


# Routes

@app.after_request
def no_store(resp):
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return ('', 204)


@app.route('/api/prices', methods=['GET'])
def api_prices():
    try:
        df = fetch_live_prices()
        return jsonify({'players': df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/entry', methods=['GET'])
def api_entry():
    entry_id = request.args.get('entry_id')
    which = request.args.get('which', 'next')
    if not entry_id:
        return jsonify({'error': 'entry_id is required'}), 400
    try:
        ev = detect_event(which)
        picks_js = fetch_entry_picks(int(entry_id), ev)
        bank_tenths = (picks_js.get('entry_history', {}) or {}).get('bank')
        if bank_tenths in (None, ''):
            info = fetch_entry_info(int(entry_id))
            bank_tenths = info.get('last_deadline_bank') or info.get('bank') or 0
        bank = (bank_tenths or 0) / 10.0
        return jsonify({'event': ev, 'bank': bank})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/suggest', methods=['POST'])
def api_suggest():
    data = request.get_json(force=True)
    entry_id = data.get('entry_id')
    event = data.get('event')
    top = int(data.get('top', 10))

    # Load predictions
    pred_path = find_latest_predictions_file()
    if not pred_path or not pred_path.exists():
        return jsonify({'error': 'No predictions file found (predictions_*.csv).'}), 400
    pred = pd.read_csv(pred_path)
    needed = {'season', 'gw', 'position', 'player', 'team', 'prediction'}
    if not needed.issubset(set(pred.columns)):
        return jsonify({'error': 'Predictions CSV missing required columns.'}), 400

    prices = fetch_live_prices()

    if entry_id:
        try:
            ev = int(event) if event else detect_event('next')
            squad, bank = build_squad_from_entry(int(entry_id), ev)
        except Exception as e:
            return jsonify({'error': f'Failed to load entry team: {e}'}), 400
    else:
        squad_rows = data.get('squad', [])
        bank = float(data.get('bank', 0))
        squad = pd.DataFrame(squad_rows)
        for col in ['player', 'team', 'position']:
            if col not in squad.columns:
                return jsonify({'error': f'Missing squad field: {col}'}), 400

    recs = suggest_single_transfers(pred, squad, prices, bank=bank, top=top)
    return jsonify({
        'predictions_file': str(pred_path),
        'count': int(len(recs)),
        'suggestions': recs.to_dict(orient='records') if len(recs) else []
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)
