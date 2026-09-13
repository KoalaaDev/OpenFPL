"""E17 pre-registered arms: ten hypotheses for why the engine's defender
picks fall short, each a research variant replayed against the shipped
engine on the same 74 gameweeks. Written before any arm ran; alpha
0.05 / 10 = 0.005 for the family.

Each entry is the ``xpts_kwargs`` forwarded to ``xpts_predict_gw``; a
``constants`` block instead patches module constants for that process
(the arm runner applies it before the replay).
"""
ARMS = {
    # H1  defenders (every position) convert below their xG and assist above
    #     their blended xA: calibrate xG90 / xA90 by the position's realised
    #     ratio, point-in-time
    "h1_xg_cal": {"tweaks": {"rates": {"calibrate_by_pos": True}}},
    # H2  a defender's goals come from set pieces, which scale less with the
    #     fixture than open play: attack scaler ** 0.5 for DEF
    "h2_def_att_exp": {"tweaks": {"def_attack_exp": 0.5}},
    # H3  conceded goals are counted while he is on the pitch, so the Poisson
    #     mean should carry E[minutes]/90, not P(plays)
    "h3_conc_emin": {"tweaks": {"conceded_exposure": "e_min"}},
    # H4  team goals are overdispersed (var/mean 1.078), so P(no goals) is a
    #     negative binomial's, not a Poisson's
    "h4_cs_nb": {"tweaks": {"cs_dispersion": 0.056}},
    # H5  DefCon actions earn BPS: put threshold crossings in the bonus fit
    "h5_bonus_dc": {"tweaks": {"rates": {"bonus_defcon": True}}},
    # H6  the measured 13% DefCon under-prediction, now on defender metrics
    "h6_defcon_113": {"rate_scale": {"defcon_cross90": 1.13}},
    # H7  market-only lambda (the clean-sheet channel is the market's to price)
    "h7_odds_10": {"odds_weight": 1.0},
    # H8  home clean sheets read 1.6 points high and away 1.8 low in the
    #     calibration: home lam_against x1.06, away x0.94
    "h8_venue": {"tweaks": {"venue_adj": 0.06}},
    # H9  defenders' attacking rates are rare events on a noisy estimate:
    #     shrink them twice as hard (12 pseudo-90s instead of 6)
    "h9_def_k12": {"tweaks": {"rates": {"k_by_pos": {"DEF": 12.0}}}},
    # H10 recent defensive form: team-model half-life 90 days instead of 180
    "h10_hl90": {"constants": {"team_model": {"HALF_LIFE_DAYS": 90.0}}},
}
