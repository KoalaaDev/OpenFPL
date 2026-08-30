"""Transfermarkt parsing and point-in-time joins.

Every case here is a defect that produced plausible-looking wrong output rather
than an error, which is the only kind this source has actually produced.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from fpl_engine.ingest import transfermarkt as tm
from fpl_engine.xpts import injury_features as inj, tm_features as tmf


# --------------------------------------------------------------- rumours --
RUMOUR_ROW = '''
<table class="items"><tbody>
<tr class="odd">
  <td class="hauptlink"><table class="inline-table"><tr>
    <td><a href="/gakpo/profil/spieler/419717">Cody Gakpo</a></td></tr>
    <tr><td>Left Winger</td></tr></table></td>
  <td><a title="Liverpool FC" href="/liverpool-fc/startseite/verein/31"><img
      src="x.png" title="Liverpool FC" /></a></td>
  <td><a title="Manchester City" href="/manchester-city/startseite/verein/281"><img
      src="y.png" title="Manchester City" /></a>
      <a title="Manchester City" href="/manchester-city/startseite/verein/281">Manchester City</a>
      <a href="/premier-league/startseite/wettbewerb/GB1">Premier League</a></td>
  <td><a href="/x">01/08/2026</a></td>
  <td>69 %</td>
</tr></tbody></table>'''


def test_the_club_title_is_read_even_though_it_precedes_an_img_tag():
    # `verein/(\\d+)"[^>]*title=` cannot reach the title, because the markup is
    # `verein/281"><img src="…" title="…"` and `[^>]*` will not cross `>`.
    rows = tm.parse(RUMOUR_ROW)
    assert len(rows) == 1
    assert rows[0]["from_club"] == "Liverpool FC"
    assert rows[0]["to_club"] == "Manchester City"
    assert rows[0]["probability"] == 69


def test_a_premier_league_destination_is_not_filed_as_leaving_the_league():
    # "Manchester City" does not CONTAIN "Man City", so substring matching
    # alone reports the one thing the feature exists to distinguish.
    teams = {"Man City": 13, "Liverpool": 12, "Nott'm Forest": 17}
    assert tm.resolve_club("Manchester City", teams) == 13
    assert tm.resolve_club("Nottingham Forest", teams) == 17
    assert tm.resolve_club("Real Madrid", teams) is None


# ----------------------------------------------------------------- money --
@pytest.mark.parametrize("text,want", [
    ("€31.90m", 31_900_000), ("€800k", 800_000), ("€1.20bn", 1_200_000_000),
    ("free transfer", 0), ("loan transfer", None), ("-", None), ("?", None),
    (None, None),
])
def test_a_missing_fee_is_not_a_free_transfer(text, want):
    assert tm.money_eur(text) == want


# ----------------------------------------------------------------- squad --
def _squad_row(name_suffix="", extra_dates="", crest_title=""):
    return f'''
<table class="items"><tbody>
<tr class="odd">
  <td class="zentriert rueckennummer"><div class=rn_nummer>30</div></td>
  <td class="posrela"><span class="wechsel-kader-wappen">{crest_title}</span>
    <table class="inline-table"><tr>
      <td><img src="p.jpg" title="Illan Meslier" /></td>
      <td class="hauptlink"><a href="/illan-meslier/profil/spieler/542586">
        Illan Meslier{name_suffix}</a></td></tr>
      <tr><td>Goalkeeper</td></tr></table></td>
  <td class="zentriert">02/03/2000 (26)</td>
  <td class="zentriert"><img title="France" /></td>
  <td class="zentriert">1,97m</td>
  <td class="zentriert">left</td>
  <td class="zentriert">09/07/2026</td>
  <td class="zentriert"><a title="Leeds United: Abl&ouml;se free transfer"
      href="/leeds-united/startseite/verein/399"><img title="Leeds United" /></a></td>
  {extra_dates}
  <td class="rechts hauptlink"><a
      href="/illan-meslier/marktwertverlauf/spieler/542586">€8.00m</a></td>
</tr></tbody></table>'''


def test_an_injured_player_is_not_dropped_from_the_squad():
    # An injured player carries an icon span straight after his name, so
    # anchoring on `</a>` silently drops exactly the rows a minutes model
    # cares most about.
    hurt = '<span title="Groin injury" class="verletzt-table">&nbsp;</span>'
    rows = tm.parse_squad(_squad_row(name_suffix=hurt))
    assert len(rows) == 1
    assert rows[0]["tm_name"] == "Illan Meslier"


def test_a_new_signings_contract_is_not_read_as_the_day_he_arrived():
    # The crest of a new signing carries `title="Joined from X; date: …"`,
    # repeating the joined date inside an attribute. Left in, it becomes the
    # second bare date and every summer signing's contract reads as expiring
    # the day he walked in.
    crest = ('<a title="Joined from Leeds United; date: 09/07/2026; '
             'fee: free transfer" href="/x"><img title="Leeds" /></a>')
    row = tm.parse_squad(_squad_row(
        crest_title=crest,
        extra_dates='<td class="zentriert">30/06/2028</td>'))[0]
    assert row["joined_date"] == "2026-07-09"
    assert row["contract_until"] == "2028-06-30"


def test_a_historical_squad_page_has_no_contract_column():
    row = tm.parse_squad(_squad_row())[0]
    assert row["joined_date"] == "2026-07-09"
    assert row["contract_until"] is None
    assert row["dob"] == "2000-03-02" and row["height_cm"] == 197
    assert row["foot"] == "left" and row["detail_position"] == "Goalkeeper"
    assert row["signed_from"] == "Leeds United"
    assert row["signed_fee_eur"] == 0 and row["market_value"] == 8_000_000


# --------------------------------------------------------------- ce apis --
def test_an_agreed_but_uncompleted_move_is_not_treated_as_history():
    payload = '''{"transfers":[
      {"dateUnformatted":"2027-01-05","upcoming":true,"fee":"€40.00m",
       "from":{"clubName":"A","href":"/a/transfers/verein/1"},
       "to":{"clubName":"B","href":"/b/transfers/verein/2"}},
      {"dateUnformatted":"2026-07-09","upcoming":false,"fee":"free transfer",
       "marketValue":"€8.00m","season":"26/27",
       "from":{"clubName":"Leeds","href":"/l/transfers/verein/399"},
       "to":{"clubName":"Arsenal","href":"/a/transfers/verein/11"}}]}'''
    rows = tm.parse_transfer_list(payload)
    assert len(rows) == 1
    assert rows[0]["transfer_date"] == "2026-07-09"
    assert rows[0]["from_club_id"] == 399 and rows[0]["to_club_id"] == 11
    assert rows[0]["fee_eur"] == 0 and rows[0]["value_eur"] == 8_000_000


def test_market_value_graph_parses_dates_and_amounts():
    payload = ('{"list":[{"y":600000,"datum_mw":"08\\/01\\/2019",'
               '"verein":"FC Lorient","age":"18"}]}')
    rows = tm.parse_market_values(payload)
    assert rows == [{"value_date": "2019-01-08", "value_eur": 600000,
                     "club": "FC Lorient", "age": 18}]


# ------------------------------------------------------------- identity --
def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    tm.init(conn)
    tm.init2(conn)
    conn.execute("CREATE TABLE player (season TEXT, player_id INTEGER, "
                 "code INTEGER, web_name TEXT, full_name TEXT, position TEXT)")
    return conn


def test_injury_spells_join_on_the_stable_code_not_the_recycled_element_id():
    # FPL reassigns element ids every summer: measured on the live database,
    # 99.7% of ids point to a different footballer one season later. Joining a
    # Transfermarkt player through `player_id` hands his injury record to
    # whoever inherited his number.
    conn = _db()
    conn.execute("INSERT INTO player VALUES ('2024-25',7,111,'A','A','MID')")
    conn.execute("INSERT INTO player VALUES ('2026-27',7,222,'B','B','MID')")
    conn.execute("INSERT INTO tm_player (tm_player_id, tm_name, player_id, "
                 "player_code) VALUES (1,'A',7,111)")
    conn.execute("INSERT INTO tm_injury (tm_player_id, from_date, injury, "
                 "until_date, days, games_missed, season_label, observed_utc) "
                 "VALUES (1,'2024-09-01','Hamstring injury','2024-10-01',30,4,"
                 "'24/25','2026-08-30T00:00:00Z')")
    sp = inj.spells(conn, "2026-27")
    assert list(sp["player_code"]) == [111]      # never 222


def test_transfermarkt_features_never_see_a_row_dated_after_the_kickoff():
    conn = _db()
    conn.execute("INSERT INTO tm_player (tm_player_id, tm_name, player_code) "
                 "VALUES (1,'A',111)")
    conn.execute("INSERT INTO tm_squad (season, tm_player_id, tm_name, dob, "
                 "height_cm, foot, detail_position, observed_utc) VALUES "
                 "('2025-26',1,'A','2000-01-01',180,'left','Left Winger','x')")
    for d, v in (("2025-01-01", 10_000_000), ("2026-01-01", 90_000_000)):
        conn.execute("INSERT INTO tm_market_value (tm_player_id, value_date, "
                     "value_eur, observed_utc) VALUES (1,?,?,'x')", (d, v))
    conn.execute("INSERT INTO tm_transfer (tm_player_id, transfer_date, "
                 "to_club_id, fee_eur, fee_text, observed_utc) "
                 "VALUES (1,'2026-01-02',11,50000000,'€50.00m','x')")
    frame = pd.DataFrame({"player_code": [111], "kick": ["2025-08-01T14:00:00Z"],
                          "season": ["2025-26"], "gw": [1], "team_id": [1],
                          "position": ["MID"]})
    out = tmf.add_features(frame, tmf.load(conn))
    # the £90m valuation and the transfer both post-date the kickoff
    assert out["tm_mv_log"].iloc[0] == pytest.approx(np.log1p(10_000_000))
    assert pd.isna(out["tm_days_since_move"].iloc[0])
    # immutable attributes are still available
    assert out["tm_age"].iloc[0] == pytest.approx(25.6, abs=0.2)
    assert out["tm_role_wide"].iloc[0] == 1.0
