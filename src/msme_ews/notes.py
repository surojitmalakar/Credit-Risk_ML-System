"""Underwriter notes and an append-only audit trail.

Notes are decision records, not model outputs, and are stored for the current
session only. The audit trail records what was analyzed and exported so a review
can see how a figure was produced; it is not a compliance-grade record store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

NOTE_DECISIONS = (
    "Pending review",
    "Approve with monitoring",
    "Approve with conditions",
    "Refer to committee",
    "Decline",
)
MAX_NOTES = 200
MAX_EVENTS = 200


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_note(
    record_key: str,
    company: str,
    period: str,
    text: str,
    decision: str = "Pending review",
    author: str = "Analyst",
    probability: float | None = None,
    risk_category: str = "Not available",
) -> dict[str, Any]:
    """Create one note record with a stable identifier."""
    cleaned = (text or "").strip()
    return {
        "note_id": _timestamp().replace(":", "").replace("-", "")[:15],
        "created_utc": _timestamp(),
        "record_key": record_key,
        "company": company,
        "period": period,
        "decision": decision if decision in NOTE_DECISIONS else "Pending review",
        "note": cleaned,
        "author": author.strip() or "Analyst",
        "probability": probability,
        "risk_category": risk_category,
    }


def validate_note(text: str, decision: str) -> str | None:
    """Return an error message when a note cannot be saved."""
    if not (text or "").strip():
        return "Enter a note before saving."
    if len(text.strip()) > 2_000:
        return "Notes are limited to 2000 characters."
    if decision not in NOTE_DECISIONS:
        return "Choose one of the listed review decisions."
    return None


def add_note(store: list[dict[str, Any]], note: dict[str, Any]) -> list[dict[str, Any]]:
    """Append a note, keeping the store bounded."""
    store.append(note)
    del store[:-MAX_NOTES]
    return store


def notes_frame(notes: Iterable[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "created_utc", "company", "period", "decision", "risk_category",
        "probability", "author", "note", "record_key", "note_id",
    ]
    if not notes:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(list(notes)).reindex(columns=columns)


def notes_csv(notes: Iterable[dict[str, Any]]) -> bytes:
    return notes_frame(notes).to_csv(index=False).encode("utf-8")


def decision_counts(notes: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = notes_frame(notes)
    if frame.empty:
        return pd.DataFrame(columns=["Decision", "Notes", "Share %"])
    counts = frame["Decision"].value_counts()
    return pd.DataFrame([
        {
            "Decision": decision,
            "Notes": int(count),
            "Share %": round(100 * count / len(frame), 1),
        }
        for decision, count in counts.items()
    ])


def audit_event(
    event: str,
    detail: str,
    dataset: str = "",
    actor: str = "Analyst",
) -> dict[str, Any]:
    """Create one audit entry."""
    return {
        "event_id": _timestamp().replace(":", "").replace("-", "")[:15],
        "recorded_utc": _timestamp(),
        "event": event,
        "detail": detail,
        "dataset": dataset,
        "actor": actor,
    }


def log_event(store: list[dict[str, Any]], event: dict[str, Any]) -> list[dict[str, Any]]:
    store.append(event)
    del store[:-MAX_EVENTS]
    return store


def audit_frame(events: Iterable[dict[str, Any]]) -> pd.DataFrame:
    columns = ["recorded_utc", "event", "detail", "dataset", "actor", "event_id"]
    if not events:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(list(events)).reindex(columns=columns).sort_values(
        "recorded_utc", ascending=False
    ).reset_index(drop=True)


def audit_csv(events: Iterable[dict[str, Any]]) -> bytes:
    return audit_frame(events).to_csv(index=False).encode("utf-8")