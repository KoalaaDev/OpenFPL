"""The HTTP surface of a planner that is on the public internet."""
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app import auth, jobs, security, services, userdata

os.environ.setdefault("FPLABS_AUTO_REFRESH", "0")

JSON = {"Content-Type": "application/json", "X-Requested-With": "fetch"}
KEY = b"unit-test-secret-key-that-is-long-enough"


@pytest.fixture()
def client(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    monkeypatch.setattr(userdata, "APP_DB_PATH", path)
    monkeypatch.setattr(auth, "_secret_cache", KEY)
    monkeypatch.setattr(security, "limiter", security.RateLimiter())
    from app.main import app
    # no context manager: startup hooks (network prewarm, scheduler) stay off
    yield TestClient(app, base_url="http://testserver")
    try:
        os.remove(path)
    except PermissionError:
        pass


def _login(client, email="owner@example.com"):
    u = userdata.upsert_user(f"sub-{email}", email, "Owner", None)
    client.cookies.set(auth.SESSION_COOKIE, auth.sign(auth.new_session(u["id"]), KEY))
    return u


def test_security_headers_and_no_store_on_api(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 200
    h = r.headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in h["content-security-policy"]
    assert h["cache-control"] == "no-store"
    assert "strict-transport-security" not in h      # plain http in tests


def test_anonymous_visitor_gets_a_session_cookie_and_no_default_entry(client):
    r = client.get("/api/auth/me")
    assert r.json()["user"] is None
    assert auth.SESSION_COOKIE in r.cookies
    payload = auth.verify(r.cookies[auth.SESSION_COOKIE], KEY)
    assert payload["uid"] is None and payload["sid"]
    assert "HttpOnly" in r.headers["set-cookie"]


def test_mutations_without_fetch_headers_are_blocked(client):
    r = client.put("/api/prefs", data=json.dumps({"entry_id": 5}),
                   headers={"Content-Type": "application/json"})
    assert r.status_code == 403
    r = client.put("/api/prefs", data="entry_id=5",
                   headers={"Content-Type": "application/x-www-form-urlencoded",
                            "X-Requested-With": "fetch"})
    assert r.status_code == 403
    r = client.put("/api/prefs", data=json.dumps({"entry_id": 5}), headers=JSON)
    assert r.status_code == 200 and r.json()["entry_id"] == 5


def test_body_over_the_cap_is_refused(client, monkeypatch):
    big = json.dumps({"drafts": [{"pad": "x" * (security.MAX_BODY_BYTES + 10)}]})
    r = client.put("/api/drafts", data=big, headers=JSON)
    assert r.status_code == 413


def test_documents_are_private_to_the_session(client):
    r = client.put("/api/drafts", data=json.dumps({"drafts": [{"id": "mine"}]}),
                   headers=JSON)
    assert r.status_code == 200
    assert client.get("/api/drafts").json()["drafts"][0]["id"] == "mine"
    client.cookies.clear()                            # a different visitor
    assert client.get("/api/drafts").json()["drafts"] == []


def test_pull_and_build_are_admin_only(client, monkeypatch):
    monkeypatch.setenv("FPLABS_ADMIN_EMAILS", "owner@example.com")
    assert client.post("/api/pull", data="{}", headers=JSON).status_code == 403
    assert client.post("/api/projections/build", data=json.dumps({"gws": [4]}),
                       headers=JSON).status_code == 403
    _login(client, "visitor@example.com")
    assert client.post("/api/pull", data="{}", headers=JSON).status_code == 403
    _login(client, "owner@example.com")
    monkeypatch.setattr(jobs, "start", lambda *a, **k: "job-x")
    monkeypatch.setattr(jobs, "running", lambda *a, **k: [])
    r = client.post("/api/pull", data="{}", headers=JSON)
    assert r.status_code == 200 and r.json()["job_id"] == "job-x"


def test_cookie_import_is_off_by_default(client, monkeypatch):
    monkeypatch.delenv("FPLABS_ALLOW_COOKIE_IMPORT", raising=False)
    r = client.post("/api/myteam/import",
                    data=json.dumps({"entry": 1, "cookie": "pl_profile=x"}),
                    headers=JSON)
    assert r.status_code == 403


def test_jobs_are_only_visible_to_their_owner(client, monkeypatch):
    calls = []

    def fake_solve(job_id, params, principal):
        calls.append((params, principal))
        return {"plans": []}

    monkeypatch.setattr(services, "run_solve", fake_solve)
    r = client.post("/api/solve", data=json.dumps({"horizon": 99, "time_limit": 10_000}),
                    headers=JSON)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    for _ in range(50):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] != "running":
            break
    assert j["status"] == "done"
    assert calls and calls[0][1].startswith("anon:")
    client.cookies.clear()
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_solve_params_are_clamped():
    p = services.clamp_solve_params({"horizon": 99, "time_limit": 99999,
                                     "keep_per_position": 1, "max_transfers": 0,
                                     "n_plans": 10, "decay": 5, "hit_cost": -3,
                                     "locked": ["7", "x", 9]})
    assert p["horizon"] == 8 and p["time_limit"] == 120
    assert p["keep_per_position"] == 8 and p["max_transfers"] == 1
    assert p["n_plans"] == 3 and p["decay"] == 1.0 and p["hit_cost"] == 0.0
    assert p["locked"] == [7, 9]
    assert services.clamp_solve_params({})["horizon"] == 5


def test_squad_document_is_validated(client):
    bad = {"squad": [{"element": "abc"}] * 15}
    r = client.put("/api/myteam", data=json.dumps(bad), headers=JSON)
    assert r.status_code == 400
    dup = {"squad": [{"element": 1, "selling_price": 4.0}] * 15}
    r = client.put("/api/myteam", data=json.dumps(dup), headers=JSON)
    assert r.status_code == 400
    ok = {"squad": [{"element": i, "selling_price": 4.0} for i in range(1, 16)],
          "bank": "1.5", "free_transfers": 99}
    r = client.put("/api/myteam", data=json.dumps(ok), headers=JSON)
    assert r.status_code == 200
    doc = r.json()
    assert doc["free_transfers"] == 5 and doc["bank"] == 1.5
    assert len(doc["squad"]) == 15


def test_entry_ids_are_bounded(client):
    assert client.get("/api/entry/0").status_code == 400
    assert client.get("/api/entry/999999999").status_code == 400


def test_rate_limiter_token_bucket():
    rl = security.RateLimiter({"heavy": (2, 1.0)})
    assert rl.allow("1.2.3.4", "heavy", now=0.0)
    assert rl.allow("1.2.3.4", "heavy", now=0.0)
    assert not rl.allow("1.2.3.4", "heavy", now=0.0)
    assert rl.allow("5.6.7.8", "heavy", now=0.0)          # another client
    assert rl.allow("1.2.3.4", "heavy", now=1.5)          # refilled one


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(security, "limiter", security.RateLimiter({"cheap": (2, 0.0)}))
    assert client.get("/api/auth/me").status_code == 200
    assert client.get("/api/auth/me").status_code == 200
    assert client.get("/api/auth/me").status_code == 429


def test_logout_issues_a_fresh_anonymous_session(client):
    u = _login(client)
    assert client.get("/api/auth/me").json()["user"]["email"] == u["email"]
    r = client.post("/api/auth/logout", data="{}", headers=JSON)
    assert r.status_code == 200
    assert client.get("/api/auth/me").json()["user"] is None


def test_docs_are_hidden_unless_debug(client):
    assert client.get("/api/docs").status_code == 404
    assert client.get("/api/openapi.json").status_code == 404


def test_legal_pages_render_with_substitutions(client, monkeypatch):
    monkeypatch.setenv("FPLABS_BASE_URL", "https://fpl.koalaa.dev")
    monkeypatch.setenv("FPLABS_CONTACT_EMAIL", "privacy@koalaa.dev")
    for path in ("/privacy", "/terms"):
        r = client.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert "{{" not in r.text
        assert "https://fpl.koalaa.dev" in r.text and "privacy@koalaa.dev" in r.text
        assert "KoalaaDev" in r.text


def test_delete_account_removes_user_and_documents(client):
    u = _login(client)
    client.put("/api/drafts", data=json.dumps({"drafts": [{"id": "d"}]}), headers=JSON)
    assert userdata.get_doc(f"user:{u['id']}", "drafts") is not None
    r = client.delete("/api/auth/me", headers=JSON)
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert userdata.get_user(u["id"]) is None
    assert userdata.get_doc(f"user:{u['id']}", "drafts") is None
    assert client.get("/api/auth/me").json()["user"] is None
    client.cookies.clear()
    assert client.delete("/api/auth/me", headers=JSON).status_code == 401


def test_ui_prefs_round_trip_and_size_cap(client):
    r = client.put("/api/prefs", data=json.dumps({"ui": {"tab": "Solver", "solver.horizon": 6}}),
                   headers=JSON)
    assert r.status_code == 200 and r.json()["ui"]["tab"] == "Solver"
    assert client.get("/api/auth/me").json()["prefs"]["ui"]["solver.horizon"] == 6
    huge = {"ui": {"blob": "x" * 20_000}}
    r = client.put("/api/prefs", data=json.dumps(huge), headers=JSON)
    assert r.status_code == 200 and "blob" not in (r.json().get("ui") or {})   # ignored, not stored


def test_deadline_desk_is_admin_only(client, monkeypatch):
    monkeypatch.setenv("FPLABS_ADMIN_EMAILS", "owner@example.com")
    assert client.get("/api/admin/deadline").status_code == 403
    _login(client, "visitor@example.com")
    assert client.get("/api/admin/deadline").status_code == 403
    _login(client, "owner@example.com")
    from app import deadline
    monkeypatch.setattr(deadline, "payload", lambda force=False: {"gw": 5, "news": []})
    r = client.get("/api/admin/deadline?force=1")
    assert r.status_code == 200 and r.json()["gw"] == 5
