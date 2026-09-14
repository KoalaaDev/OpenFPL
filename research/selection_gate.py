"""Round 21 gate — does the manager's last selection carry information the
shipped minutes model does not already have?

Builds the minutes-model frame with the `sel` block attached (the same
point-in-time builder that serves the model), joins the current baseline's
audit rows (what the shipped model said: p_start) for 2024-25 and 2025-26,
and asks per feature whether the START RESIDUAL (actual start minus the
model's P(start)) moves with it — overall and inside the 0.3-0.7 band where
the model admits it does not know. A feature that predicts the residual is
information the model is missing; one that does not is already absorbed.

    python research/selection_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_engine import config, db  # noqa: E402
from fpl_engine.xpts import minutes_model as mm, selection_features as sel  # noqa: E402

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]


def main():
    conn = db.connect(config.DB_PATH)
    mm.set_extras("sel")
    frame = mm._frame(conn, SEASONS)
    auds = []
    for season in ("2024-25", "2025-26"):
        # the audits of the model BEFORE the selection block shipped: the
        # residual gate must be against a model that does not have the feature
        a = pd.read_csv(os.path.join(config.DATA_DIR, "bt_base_pre_r21", f"audit_{season}.csv"),
                        usecols=["gw", "player_id", "p_start", "starts", "n_fixtures"])
        a["season"] = season
        auds.append(a[a["n_fixtures"] == 1])
    aud = pd.concat(auds, ignore_index=True)
    d = frame.merge(aud, on=["season", "gw", "player_id"], how="inner", suffixes=("", "_aud"))
    d["resid"] = pd.to_numeric(d["starts_aud"], errors="coerce").fillna(0) - d["p_start"]
    d["band"] = (d["p_start"] >= 0.3) & (d["p_start"] < 0.7)
    print(f"rows joined {len(d)} | in band {int(d.band.sum())}")
    from scipy import stats
    print(f"\n{'feature':<24}{'n':>7}{'coef':>9}{'t':>7}{'p':>9} |{'band n':>7}{'coef':>9}{'t':>7}{'p':>9} | notes")
    for f in sel.FEATURES:
        x = pd.to_numeric(d[f], errors="coerce")
        ok = x.notna()
        if ok.sum() < 200 or x[ok].std() == 0:
            print(f"{f:<24}{int(ok.sum()):>7}   (no variation / too few rows)")
            continue
        def fit(mask):
            xx = x[mask].to_numpy(float); yy = d.loc[mask, "resid"].to_numpy(float)
            if len(xx) < 100 or xx.std() == 0:
                return (len(xx), np.nan, np.nan, np.nan)
            r = stats.linregress(xx, yy)
            return (len(xx), r.slope, r.slope / r.stderr if r.stderr > 0 else np.nan, r.pvalue)
        n1, c1, t1, p1 = fit(ok)
        n2, c2, t2, p2 = fit(ok & d["band"])
        note = ""
        if set(x[ok].unique()) <= {0.0, 1.0}:
            m1 = d.loc[ok & (x == 1), "resid"].mean(); m0 = d.loc[ok & (x == 0), "resid"].mean()
            note = f"resid when 1: {m1:+.3f} (n={int((ok & (x == 1)).sum())}) vs 0: {m0:+.3f}"
        print(f"{f:<24}{n1:>7}{c1:>9.4f}{t1:>7.1f}{p1:>9.4f} |{n2:>7}{c2:>9.4f}{t2:>7.1f}{p2:>9.4f} | {note}")
    # the rotation-threshold question directly: residual by consecutive starts
    print("\nstart residual by consecutive starts (all rows with P(start) >= 0.5):")
    hi = d[d["p_start"] >= 0.5].copy()
    hi["cs_b"] = pd.cut(hi["consec_starts"], [-1, 0, 1, 2, 3, 5, 8, 12, 99],
                        labels=["0", "1", "2", "3", "4-5", "6-8", "9-12", "13+"])
    print(hi.groupby("cs_b", observed=True).agg(n=("resid", "size"), p_start=("p_start", "mean"),
                                                started=("starts_aud", "mean"), resid=("resid", "mean")).round(3).to_string())
    print("\n... split by congestion (matches in last 14 days >= 3):")
    hi["congest"] = hi["team_matches_14d"] >= 3
    print(hi.groupby(["congest", "cs_b"], observed=True)["resid"].agg(["size", "mean"]).round(3).to_string())
    # ---- Round 21b refinements (owner) ----------------------------------
    from scipy import stats as _st
    def _cmp(label, mask1, mask0):
        a, b = d.loc[mask1, "resid"], d.loc[mask0, "resid"]
        if len(a) < 30:
            print(f"  {label:<58} n={len(a)} (too few)"); return
        t = _st.ttest_ind(a, b, equal_var=False)
        print(f"  {label:<58} n={len(a):>6} resid {a.mean():+.3f} vs {b.mean():+.3f}  p={t.pvalue:.4f}")
    regular = d["start_rate_l10"].fillna(0) >= 0.7
    print("\nD. suspension threat by importance (one booking from a ban):")
    bt = d["ban_threat"] == 1
    _cmp("regular (start rate >= 0.7), one booking from a ban", bt & regular, (~bt) & regular)
    _cmp("fringe, one booking from a ban", bt & ~regular, (~bt) & ~regular)
    _cmp("regular on exactly 4 yellows (any matchday)", (d["yellows_todate"] == 4) & regular, (d["yellows_todate"] != 4) & regular)
    print("\nB. the specific stand-in facing the regular's return:")
    rep = d["replacement_for_returning"] == 1
    _cmp("replacement_for_returning == 1 (all)", rep, ~rep)
    _cmp("replacement_for_returning == 1, in band", rep & d["band"], (~rep) & d["band"])
    _cmp("pos_regular_returning >= 1 but NOT the slot holder", (d["pos_regular_returning"] >= 1) & ~rep, (d["pos_regular_returning"] == 0) & ~rep)
    print("\nC. home/away bias with a minimum sample (>= 8 home and 8 away rows):")
    hb = pd.to_numeric(d["home_start_bias"], errors="coerce")
    cnt = d.groupby("player_code").cumcount()
    ok = hb.notna() & (cnt >= 16)
    r = _st.linregress(hb[ok], d.loc[ok, "resid"]); print(f"  n={int(ok.sum())} coef {r.slope:+.4f} p={r.pvalue:.4f}")
    okb = ok & d["band"]
    r = _st.linregress(hb[okb], d.loc[okb, "resid"]); print(f"  band n={int(okb.sum())} coef {r.slope:+.4f} p={r.pvalue:.4f}")
    print("\nE. team shocks as MANAGER REACTION, by the player's position:")
    gd = pd.to_numeric(d["gd_l1"], errors="coerce")
    for pos in ("GK", "DEF", "MID", "FWD"):
        m = d["position"] == pos
        _cmp(f"{pos}: conceded 3+ last match", m & (d["team_ga_l1"] >= 3), m & (d["team_ga_l1"] < 3))
        _cmp(f"{pos}: lost by 3+ last match", m & (gd <= -3), m & (gd > -3))
        _cmp(f"{pos}: won by 3+ last match", m & (gd >= 3), m & (gd < 3))
        _cmp(f"{pos}: lost last match", m & (d["team_lost_l1"] == 1), m & (d["team_lost_l1"] == 0))
    # and the XI itself: how many of last match's starters are dropped, by shock
    print("\nE2. share of last match's XI retained (team level), by last result:")
    t = d.drop_duplicates(["season", "gw", "team_id"])[["xi_kept_l1", "team_ga_l1", "gd_l1", "team_lost_l1"]].dropna()
    for label, m in (("conceded 3+", t["team_ga_l1"] >= 3), ("conceded < 3", t["team_ga_l1"] < 3),
                     ("lost by 3+", t["gd_l1"] <= -3), ("won by 3+", t["gd_l1"] >= 3),
                     ("lost", t["team_lost_l1"] == 1), ("did not lose", t["team_lost_l1"] == 0)):
        print(f"  {label:<14} n={int(m.sum()):>5}  XI kept {t.loc[m, 'xi_kept_l1'].mean():.3f}")
    conn.close()


if __name__ == "__main__":
    main()
