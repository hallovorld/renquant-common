"""Canonical ntfy sender for the RenQuant fleet (compliance campaign B6, audit XC-4).

Before this module the fleet carried ~10 Python + 8 shell ntfy sender copies
across renquant-orchestrator / renquant-base-data / renquant-backtesting with
semantic drift: priority none/3/4/5, timeout 5 vs 10, and — the operational
bug — ``RENQUANT_NO_NOTIFY`` honored only OUTSIDE the orchestrator, so the
orchestrator's monitors could not be muted via the documented env. This module
is the single Python sender; ``RenQuant/scripts/notify.sh`` is the single
shell sender. Both enforce the same contract:

* **Topic resolution** (existing fleet convention, unified): explicit
  ``topic`` argument > ``NTFY_TOPIC`` env var > ``NTFY_TOPIC=`` line parsed
  from an env file (the explicit ``env_file`` argument, else ``$RQ_ROOT/.env``
  when ``RQ_ROOT`` is set) > the fleet default topic ``"renquant"``.
* **``RENQUANT_NO_NOTIFY`` honored ALWAYS**: a truthy value (``1``/``true``/
  ``yes``/``on``, case-insensitive) suppresses every send, logged at INFO.
* **Timeout standardized** at :data:`DEFAULT_TIMEOUT_SECONDS` (5 s).
* **Never raises into the caller**: any failure — network, encoding, header
  building — is swallowed, counted (:func:`send_failure_count`), and logged
  as a warning. A notification failure must never kill a monitor.

Per this repo's hard boundaries the module is stdlib-only and contains no
machine-local paths; env-file discovery goes through the ``RQ_ROOT`` env var
or an explicit argument.
"""
from __future__ import annotations

import base64
import logging
import os
import urllib.request
from pathlib import Path
from typing import Iterable, Mapping

log = logging.getLogger("renquant_common.notify")

#: Fleet default ntfy topic — consistent across every audited sender copy.
DEFAULT_TOPIC = "renquant"

#: Standardized POST timeout. The audited copies used 5 s everywhere except
#: one 10 s outlier (backtesting); 5 s is the canon.
DEFAULT_TIMEOUT_SECONDS = 5.0

_ENV_TRUTHY = frozenset({"1", "true", "yes", "on"})

_send_failures = 0


def send_failure_count() -> int:
    """Number of failed :func:`send` attempts in this process (fail-soft counter)."""
    return _send_failures


def notifications_suppressed(environ: Mapping[str, str] | None = None) -> bool:
    """Whether ``RENQUANT_NO_NOTIFY`` requests suppression (truthy, case-insensitive)."""
    env = os.environ if environ is None else environ
    return str(env.get("RENQUANT_NO_NOTIFY", "")).strip().lower() in _ENV_TRUTHY


def _topic_from_env_file(env_file: str | Path) -> str | None:
    """Parse a ``NTFY_TOPIC=`` line from an ``.env``-style file, best-effort."""
    try:
        lines = Path(env_file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw in lines:
        line = raw.strip()
        if line.startswith("NTFY_TOPIC="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    return None


def resolve_topic(
    topic: str | None = None,
    *,
    env_file: str | Path | None = None,
) -> str:
    """Resolve the ntfy topic per the fleet convention (see module docstring)."""
    if topic:
        return str(topic)
    env_topic = os.environ.get("NTFY_TOPIC", "").strip()
    if env_topic:
        return env_topic
    if env_file is None:
        rq_root = os.environ.get("RQ_ROOT", "").strip()
        if rq_root:
            env_file = Path(rq_root) / ".env"
    if env_file is not None:
        file_topic = _topic_from_env_file(env_file)
        if file_topic:
            return file_topic
    return DEFAULT_TOPIC


def encode_header(value: str) -> str:
    """Make one header value safe for the wire, without losing information.

    HTTP header values are latin-1 on the wire and ``urllib`` encodes them
    that way. A non-ASCII value therefore raises ``UnicodeEncodeError`` while
    the request is being built — which means the WHOLE notification is lost,
    body included, not merely its title.

    Measured 2026-07-29 in the live fleet log:

        ntfy send failed (failure #1 in this process,
        title='rq104 blend 假想前10 — 2026-07-28'):
        'latin-1' codec can't encode characters in position 12-14

    The daily blend readout had been composing a perfectly good alert and
    dropping it on the floor. Anything the operator writes naturally — a
    Chinese noun, an em dash, a curly quote — silently disables the alert it
    is attached to. For a fleet whose reliability work is ultimately delivered
    BY these alerts, that is a hole in the safety net rather than a cosmetic
    bug.

    ntfy accepts RFC 2047 encoded words, so ASCII passes through untouched and
    anything else is base64-wrapped and decoded by the client.
    """
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        wrapped = base64.b64encode(value.encode("utf-8")).decode("ascii")
        return f"=?UTF-8?B?{wrapped}?="


def send(
    title: str,
    body: str,
    topic: str | None = None,
    *,
    priority: int | str | None = None,
    tags: str | Iterable[str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    env_file: str | Path | None = None,
) -> bool:
    """POST one ntfy notification; the fleet's single Python sender.

    ``topic`` stays the third POSITIONAL parameter so the pre-consolidation
    ``post_ntfy(title, body, topic)`` call shape (and the injectable
    ``Callable[[str, str, str], bool]`` poster seams built on it) re-point
    without call-site rewrites.

    Returns ``True`` only when the POST was attempted and succeeded; ``False``
    when suppressed by ``RENQUANT_NO_NOTIFY`` or on any failure. NEVER raises.
    """
    global _send_failures
    try:
        if notifications_suppressed():
            log.info("[ntfy suppressed] %s: %s", title, body)
            return False
        resolved = resolve_topic(topic, env_file=env_file)
        headers = {"Title": encode_header(str(title))}
        if priority is not None:
            headers["Priority"] = str(priority).strip()
        if tags is not None:
            raw_tags = tags if isinstance(tags, str) else ",".join(str(t) for t in tags)
            headers["Tags"] = encode_header(raw_tags)
        request = urllib.request.Request(
            f"https://ntfy.sh/{resolved}",
            data=str(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(request, timeout=timeout).read()
        return True
    except Exception as exc:  # noqa: BLE001 — never raise into a monitor
        _send_failures += 1
        log.warning(
            "ntfy send failed (failure #%d in this process, title=%r): %s",
            _send_failures,
            title,
            exc,
        )
        return False


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_TOPIC",
    "notifications_suppressed",
    "resolve_topic",
    "send",
    "send_failure_count",
]
