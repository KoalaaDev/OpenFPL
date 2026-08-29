"""Polymarket quotes: parsed correctly, and kept away from the model.

Two things are being defended here.

The first is a parsing trap that produced silently wrong numbers on the first
run. Polymarket splits a three-way match into separate Yes/No markets, and the
draw's label is "Draw (Crystal Palace FC vs. Manchester City FC)" — it contains
BOTH club names. Any substring match on a team name therefore also matches the
draw, and the away side quietly inherits the draw's price. It looked plausible:
every row summed to 1.0, and only the equality of p_draw and p_away gave it
away.

The second is architectural. `odds_model.fixture_odds_map` reads every row in
`match_odds` for a fixture and lets the last one win, so a second source there
would move lambda depending on row order — a model change with no backtest
behind it. These quotes live in `market_quote` and are for display only.
"""
import pytest

from fpl_engine.ingest import polymarket as pm


def _event(home="Crystal Palace FC", away="Manchester City FC",
           ph="0.185", pd="0.235", pa="0.575"):
    """Shaped exactly as the Gamma API returns it, draw label included."""
    def leg(label, price):
        return {"groupItemTitle": label, "outcomes": '["Yes", "No"]',
                "outcomePrices": f'["{price}", "{1 - float(price):.3f}"]'}
    return {
        "title": f"{home} vs. {away}",
        "volume": "595478", "liquidity": "2763665", "slug": "cry-mci",
        "endDate": "2026-08-28T19:00:00Z",
        "markets": [
            leg(home, ph),
            leg(f"Draw ({home} vs. {away})", pd),
            leg(away, pa),
        ],
    }


# ------------------------------------------------------- the parsing trap ---
def test_the_draw_leg_does_not_steal_the_away_price():
    q = pm.parse_match_event(_event())
    assert q is not None
    # raw 0.185 / 0.235 / 0.575, normalised
    assert q["p_home"] == pytest.approx(0.185 / 0.995, abs=1e-3)
    assert q["p_draw"] == pytest.approx(0.235 / 0.995, abs=1e-3)
    assert q["p_away"] == pytest.approx(0.575 / 0.995, abs=1e-3)
    assert q["p_draw"] != pytest.approx(q["p_away"], abs=1e-6), (
        "draw and away identical is the signature of the substring bug")


def test_probabilities_are_normalised():
    q = pm.parse_match_event(_event())
    assert q["p_home"] + q["p_draw"] + q["p_away"] == pytest.approx(1.0)


def test_the_favourite_survives_parsing():
    """City at 0.575 must come out as the away favourite, not the draw."""
    q = pm.parse_match_event(_event())
    assert q["p_away"] > q["p_home"] and q["p_away"] > q["p_draw"]


def test_sub_markets_are_ignored():
    """'A vs. B - Exact Score' is a different market, not the match result."""
    ev = _event()
    ev["title"] = "Crystal Palace FC vs. Manchester City FC - Exact Score"
    assert pm.parse_match_event(ev) is None


def test_an_event_with_no_usable_legs_returns_none():
    ev = _event()
    ev["markets"] = []
    assert pm.parse_match_event(ev) is None


# ------------------------------------------------------ team resolution ----
@pytest.mark.parametrize("poly,fpl", [
    ("Manchester City FC", "Man City"),
    ("Manchester United FC", "Man Utd"),
    ("Tottenham Hotspur FC", "Spurs"),
    ("Nottingham Forest FC", "Nott'm Forest"),
    ("AFC Bournemouth", "Bournemouth"),
    ("Crystal Palace FC", "Crystal Palace"),
])
def test_club_names_resolve(poly, fpl):
    names = {fpl: 7, "Arsenal": 1, "Chelsea": 2}
    assert pm.resolve_team(poly, names) == 7


def test_an_unknown_club_resolves_to_nothing_rather_than_guessing():
    """A wrong mapping attaches one club's price to another's fixture, which is
    worse than having no price."""
    assert pm.resolve_team("Real Madrid CF", {"Arsenal": 1, "Chelsea": 2}) is None


# ------------------------------------------------------------ separation ---
def test_quotes_never_land_in_the_table_the_model_reads():
    src = open(pm.__file__, encoding="utf-8").read()
    assert "market_quote" in src
    assert "INSERT" in src and "match_odds" not in src.split('"""', 2)[2], (
        "prediction-market prices must not be written into match_odds — "
        "odds_model reads every row there and the last one wins")
