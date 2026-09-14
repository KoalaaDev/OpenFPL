"""Round 18, stage 1 (scaffold): label press-conference clauses with an LLM,
so a small encoder can be distilled from them.

Why this shape. The rules extractor (fpl_engine/pressers.py) proves the
information is there — a manager's "out" halves a likely starter's real
chance — but it is right only about half the time, because English is not a
lexicon. The lever is precision. The cheapest way to buy it is: have a large
model read each clause and emit (player, class) structured output, store the
labels, then fine-tune a small encoder on them so the Friday page is scored
offline in milliseconds. "Sentiment" is not the target; availability is.

Usage (needs ANTHROPIC_API_KEY; ~1,400 mentions a season, a few dollars):

    python research/presser_label.py --seasons 2024-25 2025-26 --limit 200

Writes `presser_label` rows (post_urn, player_id, llm_cls, llm_confidence,
rationale, model). Nothing here touches the engine; the labels are compared
with the rules classes and with realised starts by the gate script.

Classes: out | doubt | rested | available | none — the same vocabulary as
the rules pass, so the two are directly comparable and the small model is
a drop-in replacement for `pressers.classify`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_engine import config, db  # noqa: E402

MODEL = os.environ.get("FPLABS_LABEL_MODEL", "claude-sonnet-5")
SCHEMA = """
CREATE TABLE IF NOT EXISTS presser_label (
    post_urn      TEXT NOT NULL,
    player_id     INTEGER NOT NULL,
    season        TEXT NOT NULL,
    llm_cls       TEXT NOT NULL,
    llm_confidence REAL,
    rationale     TEXT,
    model         TEXT,
    labelled_utc  TEXT NOT NULL,
    PRIMARY KEY (post_urn, player_id)
);
"""

SYSTEM = (
    "You read Premier League managers' press-conference quotes for a fantasy "
    "football model. For ONE named player, classify what the text says about "
    "his availability for the club's NEXT match. Classes: out (will not play), "
    "doubt (genuinely uncertain: to be assessed, late call, 50-50), rested "
    "(fit but likely not to start), available (fit and in contention), none "
    "(the text says nothing about his availability). Answer with JSON only: "
    '{"cls": ..., "confidence": 0-1, "rationale": "<= 20 words"}.'
)


def _client():
    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit("pip install anthropic") from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set — set it in .env or the shell")
    return anthropic.Anthropic()


def label_one(client, player: str, text: str) -> dict:
    msg = client.messages.create(
        model=MODEL, max_tokens=120, system=SYSTEM,
        messages=[{"role": "user", "content": f"Player: {player}\n\nText: {text}"}])
    raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    try:
        out = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, IndexError):
        out = {"cls": "none", "confidence": 0.0, "rationale": f"unparsed: {raw[:60]}"}
    out["cls"] = out.get("cls") if out.get("cls") in ("out", "doubt", "rested", "available", "none") else "none"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=[config.CURRENT_SEASON])
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    conn = db.connect(config.DB_PATH)
    conn.executescript(SCHEMA)
    client = _client()
    rows = conn.execute(
        "SELECT o.post_urn, o.player_id, o.season, p.full_name, s.text FROM presser_obs o "
        "JOIN acq_bbc_presser s ON s.post_urn=o.post_urn "
        "JOIN player p ON p.season=o.season AND p.player_id=o.player_id "
        "WHERE o.source='presser' AND o.season IN (%s) AND NOT EXISTS ("
        "  SELECT 1 FROM presser_label l WHERE l.post_urn=o.post_urn AND l.player_id=o.player_id) "
        "GROUP BY o.post_urn, o.player_id LIMIT ?" % ",".join("?" * len(args.seasons)),
        (*args.seasons, args.limit)).fetchall()
    print(f"labelling {len(rows)} (post, player) pairs with {MODEL}")
    for i, (urn, pid, season, name, text) in enumerate(rows, 1):
        out = label_one(client, name, text[:1500])
        conn.execute(
            "INSERT OR REPLACE INTO presser_label VALUES (?,?,?,?,?,?,?,?)",
            (urn, pid, season, out["cls"], float(out.get("confidence") or 0.0),
             str(out.get("rationale") or "")[:200], MODEL,
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        if i % 25 == 0:
            conn.commit()
            print(f"  {i}/{len(rows)}")
    conn.commit()
    agree = conn.execute(
        "SELECT AVG(o.cls = l.llm_cls) FROM presser_label l JOIN presser_obs o "
        "ON o.post_urn=l.post_urn AND o.player_id=l.player_id AND o.source='presser'").fetchone()[0]
    print(f"done; rules/LLM agreement so far: {agree:.2f}" if agree is not None else "done")


if __name__ == "__main__":
    main()
