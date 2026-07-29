"""A non-ASCII title used to drop the whole notification, body included.

Measured 2026-07-29 in the live fleet log:

    ntfy send failed (failure #1 in this process,
    title='rq104 blend 假想前10 — 2026-07-28'):
    'latin-1' codec can't encode characters in position 12-14

HTTP header values go out as latin-1 and `urllib` encodes them that way, so the
request could not even be built. The daily blend readout composed a correct
alert and lost it on the floor. Anything an operator writes naturally — a
Chinese noun, an em dash, a curly quote — silently disabled the alert attached
to it, which for a fleet whose reliability work is delivered BY these alerts is
a hole in the safety net.
"""
from __future__ import annotations

import base64

import pytest

from renquant_common import notify

# The exact title from the incident.
REAL_TITLE = "rq104 blend 假想前10 — 2026-07-28"


@pytest.fixture(autouse=True)
def _unsuppressed(monkeypatch):
    monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)
    monkeypatch.setattr(notify, "resolve_topic", lambda *a, **k: "test-topic")


class _Captured:
    """Stands in for urlopen, recording the built Request."""

    def __init__(self):
        self.request = None

    def __call__(self, request, timeout=None):
        self.request = request

        class R:
            @staticmethod
            def read():
                return b"ok"
        return R()


def _send(monkeypatch, title, body="body text", **kw):
    cap = _Captured()
    monkeypatch.setattr(notify.urllib.request, "urlopen", cap)
    ok = notify.send(title, body, **kw)
    return ok, cap.request


def test_ascii_titles_are_untouched():
    assert notify.encode_header("rq104 blend top-10") == "rq104 blend top-10"


def test_the_incident_title_now_encodes():
    encoded = notify.encode_header(REAL_TITLE)
    assert encoded.startswith("=?UTF-8?B?") and encoded.endswith("?=")
    payload = encoded[len("=?UTF-8?B?"):-len("?=")]
    assert base64.b64decode(payload).decode("utf-8") == REAL_TITLE


def test_the_encoded_header_survives_latin_1(monkeypatch):
    """The actual failure mode: the header must be latin-1 encodable."""
    notify.encode_header(REAL_TITLE).encode("latin-1")  # must not raise


def test_the_incident_notification_now_SENDS(monkeypatch):
    ok, request = _send(monkeypatch, REAL_TITLE)
    assert ok is True
    assert request is not None
    # the whole thing goes out, not just a stripped title
    assert request.data == b"body text"


def test_the_body_is_not_lost_to_a_title_problem(monkeypatch):
    """What the bug really cost: the payload, not the decoration."""
    ok, request = _send(monkeypatch, "紧急", body="book is 94% cash")
    assert ok is True
    assert request.data == b"book is 94% cash"


def test_a_non_ascii_body_still_goes_as_utf8(monkeypatch):
    ok, request = _send(monkeypatch, "ascii title", body="持仓 94% 现金")
    assert ok is True
    assert request.data == "持仓 94% 现金".encode("utf-8")


def test_non_ascii_tags_are_encoded_too(monkeypatch):
    ok, request = _send(monkeypatch, "t", tags=["警告", "rq104"])
    assert ok is True
    request.get_header("Tags").encode("latin-1")  # must not raise


def test_ascii_tags_stay_readable(monkeypatch):
    ok, request = _send(monkeypatch, "t", tags=["warning", "rq104"])
    assert request.get_header("Tags") == "warning,rq104"


def test_priority_is_unaffected(monkeypatch):
    ok, request = _send(monkeypatch, "t", priority=5)
    assert request.get_header("Priority") == "5"


@pytest.mark.parametrize("title", [
    "em dash — here",          # the other half of the incident title
    "curly ’quotes’",
    "arrow → target",
    "degree 20°C",
    "emoji 🚨 alert",
])
def test_the_punctuation_an_operator_actually_types(monkeypatch, title):
    ok, request = _send(monkeypatch, title)
    assert ok is True, f"{title!r} still drops the notification"
    request.get_header("Title").encode("latin-1")


def test_suppression_still_wins(monkeypatch):
    monkeypatch.setenv("RENQUANT_NO_NOTIFY", "1")
    cap = _Captured()
    monkeypatch.setattr(notify.urllib.request, "urlopen", cap)
    assert notify.send(REAL_TITLE, "b") is False
    assert cap.request is None


def test_send_still_never_raises(monkeypatch):
    def boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(notify.urllib.request, "urlopen", boom)
    assert notify.send(REAL_TITLE, "b") is False
