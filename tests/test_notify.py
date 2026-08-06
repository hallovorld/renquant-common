"""Tests for the canonical ntfy sender (campaign B6, audit XC-4).

Contract under test: topic resolution order, RENQUANT_NO_NOTIFY suppression,
never-raises-into-caller (counted), priority/tags header mapping, and the
standardized timeout. No test touches the network — ``urllib.request.urlopen``
is always monkeypatched.
"""
from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from renquant_common import notify


class _FakeResponse:
    def read(self) -> bytes:
        return b"{}"


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    """Zero the retry backoff for EVERY test in this module.

    Retries landed 2026-08-05 (a timeout was losing alarms). Three pre-existing
    failure-path tests immediately got 3s slower each, because a failure now costs
    1s + 2s of real sleeping — a 9s tax on the suite for no added coverage. Autouse
    rather than per-test: the next failure-path test written here would otherwise pay
    it silently too.
    """
    monkeypatch.setattr(notify.time, "sleep", lambda _s: None)


@pytest.fixture()
def clean_env(monkeypatch):
    for var in ("NTFY_TOPIC", "RENQUANT_NO_NOTIFY", "RQ_ROOT"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture()
def capture(clean_env, monkeypatch):
    """Stub urlopen; record (Request, timeout) per call."""
    calls: list[tuple[urllib.request.Request, float]] = []

    def fake_urlopen(request, timeout=None):
        calls.append((request, timeout))
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


# ---------------------------------------------------------------------------
# topic resolution
# ---------------------------------------------------------------------------
def test_explicit_topic_wins(capture, monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "from-env")
    assert notify.send("t", "b", "explicit") is True
    assert capture[0][0].full_url == "https://ntfy.sh/explicit"


def test_env_var_topic(capture, monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "from-env")
    notify.send("t", "b")
    assert capture[0][0].full_url == "https://ntfy.sh/from-env"


def test_env_file_topic(capture, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('OTHER=1\nNTFY_TOPIC="from-file"\n', encoding="utf-8")
    notify.send("t", "b", env_file=env_file)
    assert capture[0][0].full_url == "https://ntfy.sh/from-file"


def test_rq_root_env_file_fallback(capture, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("NTFY_TOPIC=from-rq-root\n", encoding="utf-8")
    monkeypatch.setenv("RQ_ROOT", str(tmp_path))
    notify.send("t", "b")
    assert capture[0][0].full_url == "https://ntfy.sh/from-rq-root"


def test_default_topic(capture):
    notify.send("t", "b")
    assert capture[0][0].full_url == f"https://ntfy.sh/{notify.DEFAULT_TOPIC}"


def test_resolve_topic_missing_env_file_falls_back(clean_env, tmp_path):
    assert notify.resolve_topic(env_file=tmp_path / "nope.env") == notify.DEFAULT_TOPIC


def test_resolve_topic_single_quotes_and_blank_value(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("NTFY_TOPIC=\nNTFY_TOPIC='quoted'\n", encoding="utf-8")
    assert notify.resolve_topic(env_file=env_file) == "quoted"


# ---------------------------------------------------------------------------
# RENQUANT_NO_NOTIFY suppression — honored ALWAYS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "On"])
def test_no_notify_suppresses(capture, monkeypatch, value):
    monkeypatch.setenv("RENQUANT_NO_NOTIFY", value)
    assert notify.send("t", "b", "topic") is False
    assert capture == []


@pytest.mark.parametrize("value", ["", "0", "false", "off"])
def test_no_notify_falsey_does_not_suppress(capture, monkeypatch, value):
    monkeypatch.setenv("RENQUANT_NO_NOTIFY", value)
    assert notify.send("t", "b", "topic") is True
    assert len(capture) == 1


def test_suppression_beats_explicit_topic_and_counts_nothing(capture, monkeypatch):
    before = notify.send_failure_count()
    monkeypatch.setenv("RENQUANT_NO_NOTIFY", "1")
    assert notify.send("t", "b", "explicit", priority=5) is False
    assert capture == []
    assert notify.send_failure_count() == before


def test_notifications_suppressed_helper(clean_env, monkeypatch):
    assert notify.notifications_suppressed() is False
    monkeypatch.setenv("RENQUANT_NO_NOTIFY", "1")
    assert notify.notifications_suppressed() is True
    assert notify.notifications_suppressed({"RENQUANT_NO_NOTIFY": "true"}) is True
    assert notify.notifications_suppressed({}) is False


# ---------------------------------------------------------------------------
# never raises into the caller; failures counted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.URLError("net down"),
        OSError("socket"),
        RuntimeError("anything"),
        UnicodeEncodeError("latin-1", "✓", 0, 1, "header"),
    ],
)
def test_never_raises_and_counts(clean_env, monkeypatch, exc):
    def boom(request, timeout=None):
        raise exc

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    before = notify.send_failure_count()
    assert notify.send("t", "b", "topic") is False  # must not raise
    assert notify.send_failure_count() == before + 1


def test_failure_logged_as_warning(clean_env, monkeypatch, caplog):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout=None: (_ for _ in ()).throw(OSError("x"))
    )
    with caplog.at_level("WARNING", logger="renquant_common.notify"):
        notify.send("t", "b", "topic")
    assert any("ntfy send failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# priority / tags header mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("priority", "expected"),
    [(4, "4"), ("5", "5"), (" 3 ", "3"), ("urgent", "urgent")],
)
def test_priority_mapping(capture, priority, expected):
    notify.send("t", "b", "topic", priority=priority)
    assert capture[0][0].get_header("Priority") == expected


def test_no_priority_header_by_default(capture):
    notify.send("t", "b", "topic")
    assert capture[0][0].get_header("Priority") is None
    assert capture[0][0].get_header("Tags") is None


@pytest.mark.parametrize(
    ("tags", "expected"),
    [("warning,chart", "warning,chart"), (["warning", "chart"], "warning,chart"), ("rotating_light", "rotating_light")],
)
def test_tags_mapping(capture, tags, expected):
    notify.send("t", "b", "topic", tags=tags)
    assert capture[0][0].get_header("Tags") == expected


# ---------------------------------------------------------------------------
# transport shape: method, body, title, timeout
# ---------------------------------------------------------------------------
def test_post_shape_and_standardized_timeout(capture):
    assert notify.send("Title here", "body here", "topic") is True
    request, timeout = capture[0]
    assert request.get_method() == "POST"
    assert request.data == b"body here"
    assert request.get_header("Title") == "Title here"
    assert timeout == notify.DEFAULT_TIMEOUT_SECONDS == 5.0


def test_timeout_override(capture):
    notify.send("t", "b", "topic", timeout=2.5)
    assert capture[0][1] == 2.5


def test_positional_poster_compat(capture):
    """The reconciler seam types its poster Callable[[str, str, str], bool]."""
    poster = notify.send
    assert poster("t", "b", "topic") is True
    assert isinstance(poster("t", "b", "topic"), bool)


# ---------------------------------------------------------------------------
# transient retry (operator-reported 2026-08-05: a lost alarm)
# ---------------------------------------------------------------------------
#
# Measured on the fleet that day: `run-surface-drift` and `rq105-liveness` both logged
#     ntfy send failed (failure #1 in this process, …): The read operation timed out
# The monitors did their job, found real problems, and the pages evaporated on one
# flaky socket. This sender had exactly one attempt and no retry — so a transient
# network blip and a silenced fleet were the same observable outcome.


@pytest.fixture()
def flaky(clean_env, monkeypatch):
    """urlopen that fails `n` times with `exc`, then succeeds. Counts attempts."""
    state = {"calls": 0}

    def make(fail_times: int, exc: BaseException):
        def fake_urlopen(request, timeout=None):
            state["calls"] += 1
            if state["calls"] <= fail_times:
                raise exc
            return _FakeResponse()
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return state

    return make


def test_a_timeout_is_RETRIED_and_the_alarm_survives(flaky):
    """The operator's report, inverted: two timeouts must not lose the page."""
    state = flaky(2, TimeoutError("The read operation timed out"))
    assert notify.send("T", "B", "topic") is True
    assert state["calls"] == 3


def test_retries_are_BOUNDED(flaky):
    """A dead network must not hang a monitor that is itself on a schedule."""
    state = flaky(99, TimeoutError("The read operation timed out"))
    assert notify.send("T", "B", "topic") is False
    assert state["calls"] == notify.SEND_ATTEMPTS


def test_a_5xx_is_transient_and_retried(flaky):
    state = flaky(1, urllib.error.HTTPError("u", 503, "busy", {}, None))
    assert notify.send("T", "B", "topic") is True
    assert state["calls"] == 2


def test_a_4xx_is_NOT_retried(flaky):
    """The line that makes this a fix rather than a hammer: a malformed request, a bad
    topic or a rejected header is wrong in a way repetition cannot mend, and retrying
    turns one useless send into three."""
    state = flaky(99, urllib.error.HTTPError("u", 400, "bad request", {}, None))
    assert notify.send("T", "B", "topic") is False
    assert state["calls"] == 1


def test_the_happy_path_adds_NO_attempts(flaky):
    """Anti-vacuity: retries must not become latency on the common case."""
    state = flaky(0, TimeoutError())
    assert notify.send("T", "B", "topic") is True
    assert state["calls"] == 1


def test_a_recovered_send_SAYS_it_recovered(flaky, caplog):
    """A silent recovery hides how flaky the path is; the next person sizing the
    problem would measure zero."""
    import logging
    caplog.set_level(logging.WARNING)
    flaky(1, TimeoutError("boom"))
    assert notify.send("T", "B", "topic") is True
    assert any("succeeded on attempt 2" in r.getMessage() for r in caplog.records)
