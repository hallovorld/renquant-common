# ntfy bodies are capped so no alert ever arrives as an attachment   (PR #TBD)

STATUS:    delivered — one cap at the fleet's single Python send site.
WHAT:      `renquant_common.notify.send` now caps the POSTed body at
           `MAX_BODY_BYTES = 3800` UTF-8 bytes via a new pure helper
           `truncate_body(body, limit)`: a boundary-safe cut (never
           mid-codepoint, so Chinese/emoji bodies stay valid UTF-8) plus a
           marker that states exactly how many bytes were dropped and where
           the full text lives (the sender's own log). A body under the cap
           is returned byte-identical — the cap never rewrites an alert that
           fits. A capped send logs a WARNING naming the title and original
           size, so the fact of truncation is itself observable. Both are
           exported; `liveness_common.alert` and every other caller change
           nothing.
WHY/DIR:   The operator asked (2026-09-12) for the ntfy stream to be cleaned
           up and for "the ones that send attachments" to be removed. Nothing
           in the fleet attaches a file on purpose — grep for `Attach:`,
           `Filename:`, `--upload-file`, `--data-binary @` finds nothing.
           The attachments are ntfy.sh itself: any body over its 4096-byte
           `message-size-limit` is stored as a `.txt` attachment, so the phone
           shows a file icon instead of the alert text. Every Python sender
           (all the rq104/rq105 sentinels via `liveness_common.alert`, the
           monitors, the retrain chain) funnels through this one function, so
           this is the only place a cap makes auto-attachments impossible
           fleet-wide. Direction: G-A stop noise / G-D ops truth — an alert
           the operator has to tap through to read is an alert that gets
           ignored.
EVIDENCE:  artifact:      `RenQuant/logs/rq104/launchd_degradation_sentinel.out` — the `rq104 DEGRADED: 2 issue(s)` body posted on 2026-09-11 measures **4,617 bytes** (the `launchd job(s) with nonzero last exit …` paragraph listing every job's ack text), i.e. over ntfy's 4,096-byte limit → delivered as an attachment [VERIFIED — measured from the evidence log 2026-09-12; ntfy.sh 12h retention means the server itself could not be used as the source]
           prod or exp:   shared library, no production path written; behaviour change is confined to bodies over 3,800 bytes, which are truncated with a marker rather than converted to attachments by the server
           existing data: `tests/test_notify.py` + `tests/test_notify_non_ascii_title.py` **62 passed** (6 new: over-cap body lands under 4,096; under-cap body byte-identical; the measured offender's shape — sixty `com.renquant.x (last exit 1) [ack…]` entries, >4,096 bytes — lands under 4,096; a 18,000-byte Chinese body truncates without a broken codepoint; kept + dropped bytes add up exactly; the helper is idempotent and pure) [VERIFIED — 2026-09-12, umbrella venv, PYTHONPATH=src]
           best-known?:   n/a — delivery plumbing, no model claim
           scope:         "this caps the body at one send site; it does not change which alerts fire, how often, their titles, priorities or topics — the volume question is separate and addressed in the progress note below"
NEXT:      (1) Landing needs the codex merge gate, out on quota until
           2026-10-03; nothing merges before then without an operator
           decision. (2) Even merged, production consumes renquant-common
           from the pinned subrepo checkout, so the cap reaches the running
           sentinels only after a `subrepos.lock.json` pin advance +
           runtime sync — `merged is not deployed`. (3) The shell senders
           (`RenQuant/scripts/notify.sh::rq_notify` and the 17 bare-curl
           scripts) have no cap either; they live in the umbrella, which this
           repo does not write to — follow-up PR there, same shape. (4) Volume:
           on a session day during the current sell-only outage the operator
           receives ~15–20 pages that are ONE root cause (the served artifact
           cannot admit buys) echoed by ~10 independent monitors; the honest
           reduction is upstream-known suppression or a per-title daily
           cooldown at this same send site, which is a policy choice to put
           in front of the operator rather than decide here.
