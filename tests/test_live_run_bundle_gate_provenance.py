"""GOAL-5 AC6 R4, schema half: the contract now KNOWS about gate provenance.

Measured on `origin/main` before this change `[本次实测 2026-07-31]`: a bundle carrying
`wf_gate_provenance: "not-a-dict-at-all"` validated CLEAN, and the field was dropped
entirely — `model_extra` was `None`, because pydantic's default is `extra="ignore"`.
So the orchestrator's daily bundle could carry the block (orch#685) while the contract
verdict said `ok: True` knowing nothing about it.

The field is OPTIONAL. Requiring it today would fail every producer except that one
daily bundle, and a contract that rejects the runs it describes gets switched off rather
than satisfied. `require_gate_provenance=True` is the per-caller switch for when
producers have caught up.
"""

from __future__ import annotations

import pytest

from renquant_common.contracts.schemas import (
    GATE_PROVENANCE_STATUSES,
    LiveRunBundle,
    validate_live_run_bundle,
)

BASE = {
    "schema_version": 1,
    "source": "daily",
    "decision_trace": (),
    "order_intents": (),
    "submitted_orders": ({"id": "o1"},),
}


def _b(**kw):
    return dict(BASE, **kw)


def test_absence_is_still_valid_so_no_producer_breaks():
    """The non-breaking half. Every existing producer keeps validating."""
    assert validate_live_run_bundle(_b()).wf_gate_provenance is None


def test_a_present_block_is_RETAINED_not_silently_dropped():
    """The defect this closes: `extra="ignore"` discarded the block entirely."""
    block = {"status": "present", "operator_authorized_override": True,
             "override_reason": "operator accepted a documented risk"}
    got = validate_live_run_bundle(_b(wf_gate_provenance=block)).wf_gate_provenance
    assert got == block
    assert "wf_gate_provenance" in LiveRunBundle.model_fields


def test_a_NON_DICT_block_is_rejected():
    """Before: accepted and dropped. A contract that takes any shape enforces none."""
    with pytest.raises(Exception):
        validate_live_run_bundle(_b(wf_gate_provenance="not-a-dict-at-all"))


def test_an_UNKNOWN_status_is_rejected():
    with pytest.raises(ValueError):
        validate_live_run_bundle(_b(wf_gate_provenance={"status": "made-up"}))


def test_a_block_with_NO_status_is_rejected():
    """A block that cannot say which of the three situations it is records nothing."""
    with pytest.raises(ValueError):
        validate_live_run_bundle(_b(wf_gate_provenance={"passed": True}))


@pytest.mark.parametrize("status", sorted(GATE_PROVENANCE_STATUSES))
def test_all_three_statuses_are_accepted(status):
    """'No artifact' and 'artifact without a stamp' are DIFFERENT statuses, not one
    missing field — they have different remedies."""
    assert validate_live_run_bundle(
        _b(wf_gate_provenance={"status": status})).wf_gate_provenance["status"] == status


def test_the_binding_switch_defaults_OFF_and_works_when_ON():
    """AC6 R4's requirement flip, per caller."""
    assert validate_live_run_bundle(_b()) is not None            # default: off
    with pytest.raises(ValueError, match="requires it"):
        validate_live_run_bundle(_b(), require_gate_provenance=True)
    assert validate_live_run_bundle(
        _b(wf_gate_provenance={"status": "present"}), require_gate_provenance=True)


def test_an_already_built_model_still_honours_the_switch():
    """`validate_live_run_bundle` short-circuits on a LiveRunBundle instance; the
    requirement must not be skippable by handing it a model instead of a dict."""
    model = LiveRunBundle.model_validate(_b())
    with pytest.raises(ValueError, match="requires it"):
        validate_live_run_bundle(model, require_gate_provenance=True)
