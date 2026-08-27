"""Validation for acquired data. Hard failures block publication.

The rule that matters most is temporal: no observation may be dated after the
moment a decision would have been made with it, and no observation may claim
to have been published in the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Finding:
    level: str
    check: str
    detail: str
    count: int = 0


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self):
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self):
        return [f for f in self.findings if f.level == "warning"]

    def ok(self) -> bool:
        return not self.errors


def _one(conn, sql, params=()):
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def run(conn) -> Report:
    r = Report()
    err = lambda c, d, n=0: r.findings.append(Finding("error", c, d, n))
    warn = lambda c, d, n=0: r.findings.append(Finding("warning", c, d, n))

    n = _one(conn, "SELECT COUNT(*) FROM acq_player_availability "
                   "WHERE observed_utc > strftime('%Y-%m-%dT%H:%M:%SZ','now','+1 day')")
    if n:
        err("temporal.observed_not_in_future",
            "observations timestamped in the future", n)

    n = _one(conn, "SELECT COUNT(*) FROM acq_player_availability "
                   "WHERE source_published_utc IS NOT NULL "
                   "AND source_published_utc > observed_utc")
    if n:
        err("temporal.published_before_observed",
            "the source claims to have published an item after we observed it, "
            "which means one of the two clocks is wrong", n)

    n = _one(conn, "SELECT COUNT(*) FROM acq_player_availability "
                   "WHERE chance_next IS NOT NULL "
                   "AND (chance_next < 0 OR chance_next > 1)")
    if n:
        err("values.chance_is_a_probability",
            "chance_next outside [0,1] — a percentage has leaked in unscaled", n)

    n = _one(conn, "SELECT COUNT(*) FROM acq_player_availability "
                   "WHERE status IS NOT NULL "
                   "AND status NOT IN ('a','d','i','s','u','n')")
    if n:
        err("values.status_vocabulary", "unrecognised availability status", n)

    n = _one(conn, "SELECT COUNT(*) FROM acq_raw_document "
                   "WHERE content_hash IS NULL OR content_hash=''")
    if n:
        err("provenance.hash_present",
            "raw documents without a content hash cannot be deduplicated "
            "or traced", n)

    n = _one(conn, "SELECT COUNT(*) FROM acq_player_availability a "
                   "LEFT JOIN acq_source s ON s.source_id=a.source_id "
                   "WHERE s.source_id IS NULL")
    if n:
        err("provenance.source_registered",
            "observations from a source that is not in the registry", n)

    n = _one(conn, "SELECT COUNT(*) FROM acq_player_availability a "
                   "LEFT JOIN player p ON p.season=a.season "
                   "AND p.player_id=a.player_id WHERE p.player_id IS NULL")
    if n:
        warn("identity.resolves_to_player",
             "availability rows whose player_id is not in that season's squad "
             "list (expected for players removed from the game mid-season)", n)
    return r


def format_report(r: Report) -> str:
    if not r.findings:
        return "  all acquisition invariants hold"
    out = []
    for f in r.findings:
        tag = "ERROR  " if f.level == "error" else "warning"
        n = f" [{f.count}]" if f.count else ""
        out.append(f"  {tag} {f.check}{n}\n           {f.detail}")
    return "\n".join(out)
