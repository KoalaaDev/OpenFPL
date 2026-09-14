"""Sessions, Google login plumbing and per-user documents."""
import os
import tempfile
import time

import pytest

from app import auth, plans, userdata


@pytest.fixture()
def app_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    monkeypatch.setattr(userdata, "APP_DB_PATH", path)
    yield path
    try:
        os.remove(path)
    except PermissionError:
        pass


KEY = b"unit-test-secret-key-that-is-long-enough"


def test_signed_token_round_trip_and_tamper_rejection():
    tok = auth.sign({"sid": "abc", "exp": time.time() + 60}, KEY)
    assert auth.verify(tok, KEY)["sid"] == "abc"
    body, sig = tok.rsplit(".", 1)
    assert auth.verify(body + ".AAAA", KEY) is None            # bad signature
    assert auth.verify(tok, b"another-key") is None           # wrong key
    assert auth.verify(None, KEY) is None
    assert auth.verify("garbage", KEY) is None


def test_expired_token_is_rejected():
    tok = auth.sign({"sid": "abc", "exp": time.time() - 1}, KEY)
    assert auth.verify(tok, KEY) is None


def test_principal_distinguishes_anonymous_from_account():
    anon = auth.new_session()
    assert auth.principal_of(anon).startswith("anon:")
    user = auth.new_session("u1")
    assert auth.principal_of(user) == "user:u1"
    assert anon["sid"] != auth.new_session()["sid"]


def test_admin_allowlist_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("FPLABS_ADMIN_EMAILS", "Owner@Example.com, second@x.io")
    assert auth.is_admin({"email": "owner@example.com"})
    assert auth.is_admin({"email": "SECOND@X.IO"})
    assert not auth.is_admin({"email": "visitor@example.com"})
    assert not auth.is_admin(None)


def test_start_login_never_open_redirects(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "sec")
    monkeypatch.setenv("FPLABS_BASE_URL", "https://fplabs.test")
    monkeypatch.setattr(auth, "_secret_cache", KEY)
    url, cookie = auth.start_login("//evil.example")
    st = auth.verify(cookie, KEY)
    assert st["next"] == "/"
    assert "redirect_uri=https%3A%2F%2Ffplabs.test%2Fapi%2Fauth%2Fgoogle%2Fcallback" in url
    assert "client_id=cid" in url and f"state={st['state']}" in url
    _, cookie2 = auth.start_login("/planner")
    assert auth.verify(cookie2, KEY)["next"] == "/planner"


def test_finish_login_checks_state_nonce_audience(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "sec")
    monkeypatch.setenv("FPLABS_BASE_URL", "https://fplabs.test")
    monkeypatch.setattr(auth, "_secret_cache", KEY)
    _, cookie = auth.start_login("/")
    st = auth.verify(cookie, KEY)

    def exchange(code):
        assert code == "the-code"
        return {"id_token": "jwt"}

    good = {"aud": "cid", "iss": "https://accounts.google.com",
            "nonce": st["nonce"], "email_verified": "true",
            "sub": "123", "email": "a@b.c", "name": "A"}

    def verify_ok(idt, nonce):
        return auth.verify_id_token(idt, nonce, fetch=lambda url: dict(good))

    claims, nxt = auth.finish_login("the-code", cookie, st["state"],
                                    exchange=exchange, verify_token=verify_ok)
    assert claims["sub"] == "123" and nxt == "/"

    with pytest.raises(ValueError):                     # state mismatch
        auth.finish_login("the-code", cookie, "wrong", exchange=exchange,
                          verify_token=verify_ok)
    bad_aud = dict(good, aud="someone-else")
    with pytest.raises(ValueError):
        auth.verify_id_token("jwt", st["nonce"], fetch=lambda u: bad_aud)
    bad_nonce = dict(good, nonce="x")
    with pytest.raises(ValueError):
        auth.verify_id_token("jwt", st["nonce"], fetch=lambda u: bad_nonce)
    unverified = dict(good, email_verified="false")
    with pytest.raises(ValueError):
        auth.verify_id_token("jwt", st["nonce"], fetch=lambda u: unverified)


def test_docs_are_scoped_and_migrate_on_login(app_db):
    userdata.set_doc("anon:s1", "drafts", {"drafts": [{"id": "d1"}]})
    userdata.set_doc("anon:s1", "prefs", {"entry_id": 42})
    userdata.set_doc("anon:s2", "drafts", {"drafts": [{"id": "other"}]})
    assert userdata.get_doc("anon:s2", "drafts")["drafts"][0]["id"] == "other"
    assert userdata.get_doc("anon:s3", "drafts") is None

    u = userdata.upsert_user("g-sub", "a@b.c", "A", None)
    p = f"user:{u['id']}"
    userdata.set_doc(p, "prefs", {"entry_id": 7})        # the account's own
    moved = userdata.migrate("anon:s1", p)
    assert moved == ["drafts"]
    assert userdata.get_doc(p, "drafts")["drafts"][0]["id"] == "d1"
    assert userdata.get_doc(p, "prefs")["entry_id"] == 7  # not clobbered
    assert userdata.get_doc("anon:s1", "drafts") is None
    # a second login refreshes, never duplicates
    again = userdata.upsert_user("g-sub", "a@b.c", "A2", "pic")
    assert again["id"] == u["id"] and again["name"] == "A2"
    assert userdata.count_users() == 1


def test_unknown_doc_kind_is_refused(app_db):
    with pytest.raises(ValueError):
        userdata.set_doc("anon:x", "passwords", {})


def test_plans_default_to_pro_until_enforced(monkeypatch):
    monkeypatch.delenv("FPLABS_ENFORCE_PLANS", raising=False)
    assert plans.plan_for({"plan": "free"}) == "pro"
    monkeypatch.setenv("FPLABS_ENFORCE_PLANS", "1")
    assert plans.plan_for({"plan": "free"}) == "free"
    assert plans.plan_for(None) == "free"
    ent = plans.entitlements("free")
    p, clamped = plans.check(ent, {"horizon": 8, "n_plans": 3,
                                   "chips": {"wildcard": {"enabled": True}},
                                   "time_limit": 120})
    assert p["horizon"] == 3 and p["n_plans"] == 1 and p["chips"] == {}
    assert set(clamped) == {"horizon", "playstyles", "chips", "time_limit"}
    p2, clamped2 = plans.check(plans.entitlements("pro"), {"horizon": 8})
    assert p2["horizon"] == 8 and clamped2 == []


def test_dotenv_loader_never_overrides_the_environment(tmp_path, monkeypatch):
    from app.__main__ import load_dotenv
    env = tmp_path / ".env"
    env.write_text('# comment\nGOOGLE_CLIENT_ID="abc.apps"\nFPLABS_PORT=9999\nBROKEN LINE\n\nQUOTED=\'x y\'\n')
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.setenv("FPLABS_PORT", "8410")
    monkeypatch.delenv("QUOTED", raising=False)
    loaded = load_dotenv(str(env))
    assert sorted(loaded) == ["GOOGLE_CLIENT_ID", "QUOTED"]
    assert os.environ["GOOGLE_CLIENT_ID"] == "abc.apps"
    assert os.environ["FPLABS_PORT"] == "8410"          # shell wins
    assert os.environ["QUOTED"] == "x y"
    assert load_dotenv(str(tmp_path / "missing.env")) == []
