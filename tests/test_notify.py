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
