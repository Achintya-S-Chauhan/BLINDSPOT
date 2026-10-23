"""
BLINDSPOT — Activity Intelligence and Diagnostics Module

Transforms raw desktop activity events and DesktopContext records into
structured session timelines, application time summaries, and diagnostic warnings.

Design Principles:
- Pure Python logic with zero LLM dependency.
- Resilient: Handles missing fields, bad timestamps, duplicates, and empty inputs safely.
- Interoperable: Works with DesktopContext, ContextRecord, or plain JSON/dict events.
- Beginner-friendly: Small, modular functions with clear docstrings and predictable returns.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import re
import time


def format_duration_seconds(seconds: float) -> str:
    """Format duration in seconds into a friendly human-readable string."""
    sec = max(0, int(round(seconds)))
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    s = sec % 60

    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {s}s" if s > 0 else f"{minutes}m"
    return f"{s}s"


def extract_app_name(window_title: str) -> str:
    """
    Heuristically extract the primary application name from a window title.
    Examples:
      'main.py - BLINDSPOT - Visual Studio Code' -> 'Visual Studio Code'
      'Google Chrome - GitHub' -> 'Google Chrome'
      'Untitled - Notepad' -> 'Notepad'
    """
    if not window_title or not isinstance(window_title, str):
        return "Unknown"

    cleaned = window_title.strip()
    if not cleaned or cleaned.lower() in ("none", "unknown", "no active window detected"):
        return "Unknown"

    # Known common desktop application substrings (fast match)
    known_apps = [
        ("visual studio code", "Visual Studio Code"),
        ("vscode", "Visual Studio Code"),
        ("google chrome", "Google Chrome"),
        ("chrome", "Google Chrome"),
        ("firefox", "Mozilla Firefox"),
        ("microsoft edge", "Microsoft Edge"),
        ("edge", "Microsoft Edge"),
        ("slack", "Slack"),
        ("discord", "Discord"),
        ("spotify", "Spotify"),
        ("notepad", "Notepad"),
        ("calculator", "Calculator"),
        ("powershell", "PowerShell"),
        ("cmd", "Command Prompt"),
        ("terminal", "Windows Terminal"),
        ("figma", "Figma"),
        ("notion", "Notion"),
        ("postman", "Postman"),
    ]

    lower = cleaned.lower()
    for pattern, canonical in known_apps:
        if pattern in lower:
            return canonical

    # If delimiter exists (e.g. 'File - Window - App'), usually the last segment is the app
    if " - " in cleaned:
        segments = [s.strip() for s in cleaned.split(" - ") if s.strip()]
        if len(segments) > 1:
            return segments[-1]

    if " — " in cleaned:
        segments = [s.strip() for s in cleaned.split(" — ") if s.strip()]
        if len(segments) > 1:
            return segments[-1]

    return cleaned.split()[0] if cleaned.split() else "Unknown"


def normalize_event(event: Any) -> Dict[str, Any]:
    """
    Convert DesktopContext, ContextRecord, or dictionary into a standard dictionary.
    """
    if event is None:
        return {}

    # If it is a DesktopContext object
    if hasattr(event, "observation") and hasattr(event, "interpretation"):
        win = getattr(event, "window_title", "")
        return {
            "application": extract_app_name(win),
            "window_title": win,
            "activity": getattr(event, "activity", "using an application"),
            "timestamp": getattr(event, "timestamp", time.time()),
            "ocr_text": getattr(event, "ocr_text", ""),
            "confidence": getattr(event, "confidence", 1),
        }

    # If it is a ContextRecord object
    if hasattr(event, "context") and hasattr(event, "start_time"):
        win = getattr(event, "window_title", "")
        return {
            "application": extract_app_name(win),
            "window_title": win,
            "activity": getattr(event, "activity", "using an application"),
            "timestamp": getattr(event, "start_time", time.time()),
            "end_time": getattr(event, "end_time", None),
            "duration": getattr(event, "duration", 0.0),
            "ocr_text": getattr(event.context, "ocr_text", "") if hasattr(event, "context") else "",
            "confidence": getattr(event, "confidence", 1),
        }

    # If it is already a dict
    if isinstance(event, dict):
        out = dict(event)
        # Ensure application is populated
        if not out.get("application"):
            extracted = extract_app_name(out.get("window_title", ""))
            if extracted != "Unknown":
                out["application"] = extracted
            else:
                out["application"] = ""
        return out

    # Fallback generic object with attributes
    win = getattr(event, "window_title", "")
    extracted = extract_app_name(win)
    return {
        "application": getattr(event, "application", "") or (extracted if extracted != "Unknown" else ""),
        "window_title": win,
        "activity": getattr(event, "activity", "using an application"),
        "timestamp": getattr(event, "timestamp", None),
        "ocr_text": getattr(event, "ocr_text", ""),
        "confidence": getattr(event, "confidence", 1),
    }


def validate_event(
    event: Any,
    index: Optional[int] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate an individual activity event.
    Returns:
        (is_valid, sanitized_event, diagnostic_warnings)
    """
    idx_str = f"at index {index}" if index is not None else ""
    warnings: List[Dict[str, Any]] = []

    if event is None or (not isinstance(event, dict) and not hasattr(event, "__dict__")):
        return False, None, [{
            "severity": "warning",
            "code": "INVALID_EVENT",
            "message": f"Malformed event {idx_str}: event must be a dictionary or object.",
            "event_index": index,
        }]

    raw = normalize_event(event)

    # 1. Validate application / window title
    app = raw.get("application") or ""
    win = raw.get("window_title") or ""
    app_str = str(app).strip()
    win_str = str(win).strip()
    if (not app_str or app_str == "Unknown") and (not win_str or win_str == "Unknown"):
        warnings.append({
            "severity": "warning",
            "code": "MISSING_IDENTIFIER",
            "message": f"Event {idx_str} is missing both 'application' and 'window_title'.",
            "event_index": index,
        })
        return False, None, warnings

    # 2. Validate timestamp
    ts = raw.get("timestamp")
    if ts is None:
        warnings.append({
            "severity": "warning",
            "code": "MISSING_TIMESTAMP",
            "message": f"Event {idx_str} is missing a timestamp.",
            "event_index": index,
        })
        return False, None, warnings

    try:
        ts_float = float(ts)
        if ts_float <= 0:
            warnings.append({
                "severity": "warning",
                "code": "INVALID_TIMESTAMP",
                "message": f"Event {idx_str} has a non-positive timestamp ({ts}).",
                "event_index": index,
            })
            return False, None, warnings
    except (ValueError, TypeError):
        warnings.append({
            "severity": "warning",
            "code": "MALFORMED_TIMESTAMP",
            "message": f"Event {idx_str} timestamp could not be converted to float: {ts}.",
            "event_index": index,
        })
        return False, None, warnings

    # 3. Check for empty OCR diagnostic notice (heuristic)
    ocr_text = raw.get("ocr_text", "")
    if isinstance(ocr_text, str) and not ocr_text.strip():
        # Certain applications expected to have text
        text_heavy_apps = ("Visual Studio Code", "Google Chrome", "Mozilla Firefox", "Notion", "Notepad")
        if app in text_heavy_apps:
            warnings.append({
                "severity": "info",
                "code": "EMPTY_OCR",
                "message": f"Event {idx_str} ({app}): In-memory OCR buffer was empty.",
                "event_index": index,
            })

    sanitized = {
        "application": str(app).strip() or extract_app_name(win),
        "window_title": str(win).strip(),
        "activity": str(raw.get("activity", "using an application")).strip() or "using an application",
        "timestamp": ts_float,
        "ocr_text": str(ocr_text) if ocr_text is not None else "",
        "confidence": int(raw.get("confidence", 1)),
    }

    if "end_time" in raw and raw["end_time"] is not None:
        try:
            sanitized["end_time"] = float(raw["end_time"])
        except (ValueError, TypeError):
            pass

    return True, sanitized, warnings


def calculate_sessions(
    events: List[Any],
    min_event_duration: float = 5.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Group consecutive activity events by application and compute session durations.
    Handles duplicate transitions and out-of-order timestamps safely.

    Returns:
        (sessions, diagnostics)
    """
    if not events:
        return [], []

    diagnostics: List[Dict[str, Any]] = []
    valid_events: List[Dict[str, Any]] = []

    # Validate each incoming event
    for idx, ev in enumerate(events):
        ok, clean, warns = validate_event(ev, index=idx)
        diagnostics.extend(warns)
        if ok and clean:
            valid_events.append(clean)

    if not valid_events:
        return [], diagnostics

    # Check for out-of-order timestamps
    for i in range(1, len(valid_events)):
        if valid_events[i]["timestamp"] < valid_events[i - 1]["timestamp"]:
            diagnostics.append({
                "severity": "warning",
                "code": "OUT_OF_ORDER_TIMESTAMPS",
                "message": (
                    f"Timestamp retrograde: event at index {i} "
                    f"({valid_events[i]['timestamp']}) is earlier than previous event "
                    f"({valid_events[i - 1]['timestamp']})."
                ),
                "event_index": i,
            })

    # Sort valid events chronologically by timestamp
    sorted_events = sorted(valid_events, key=lambda x: x["timestamp"])

    sessions: List[Dict[str, Any]] = []
    current_session: Optional[Dict[str, Any]] = None

    for i, ev in enumerate(sorted_events):
        app = ev["application"]
        act = ev["activity"]
        ts = ev["timestamp"]
        explicit_end = ev.get("end_time")

        # Duplicate check: check against immediate predecessor
        if i > 0:
            prev = sorted_events[i - 1]
            if (
                prev["application"] == app
                and prev["window_title"] == ev["window_title"]
                and abs(prev["timestamp"] - ts) < 0.001
            ):
                diagnostics.append({
                    "severity": "info",
                    "code": "SUSPICIOUS_DUPLICATE",
                    "message": f"Duplicate event for '{app}' at timestamp {ts:.2f}.",
                    "event_index": i,
                })
                # Skip duplicate processing
                continue

        # If current session continues with the same application
        if current_session is not None and current_session["application"] == app:
            current_session["event_count"] += 1
            if ev["window_title"] and ev["window_title"] not in current_session["window_titles"]:
                current_session["window_titles"].append(ev["window_title"])

            # If explicit end_time provided on the event, extend session
            if explicit_end is not None:
                current_session["has_explicit_end"] = True
                if explicit_end > current_session["end_time"]:
                    current_session["end_time"] = explicit_end
            elif ts > current_session["end_time"]:
                current_session["end_time"] = ts

        else:
            # Close previous session if exists
            if current_session is not None:
                # If explicit end_time was not set, transition occurred at current event's timestamp
                if not current_session.get("has_explicit_end"):
                    current_session["end_time"] = ts

                duration = max(min_event_duration, current_session["end_time"] - current_session["start_time"])
                current_session["duration_seconds"] = duration
                current_session["duration_formatted"] = format_duration_seconds(duration)
                # Clean internal flag
                current_session.pop("has_explicit_end", None)
                sessions.append(current_session)

            # Start new session
            init_end = explicit_end if explicit_end is not None else ts
            current_session = {
                "application": app,
                "activity": act,
                "start_time": ts,
                "end_time": init_end,
                "duration_seconds": 0.0,
                "duration_formatted": "0s",
                "event_count": 1,
                "has_explicit_end": explicit_end is not None,
                "window_titles": [ev["window_title"]] if ev.get("window_title") else [],
            }

    # Finalize the last open session
    if current_session is not None:
        duration = max(min_event_duration, current_session["end_time"] - current_session["start_time"])
        current_session["duration_seconds"] = duration
        current_session["duration_formatted"] = format_duration_seconds(duration)
        current_session.pop("has_explicit_end", None)
        sessions.append(current_session)

    return sessions, diagnostics


def summarize_activity(
    sessions: List[Dict[str, Any]],
    valid_events_count: int = 0,
    invalid_events_count: int = 0,
) -> Dict[str, Any]:
    """
    Calculate aggregate time per application and generate a clear, human-readable summary.
    """
    if not sessions:
        return {
            "total_time_seconds": 0.0,
            "total_time_formatted": "0s",
            "total_sessions": 0,
            "valid_events_count": valid_events_count,
            "invalid_events_count": invalid_events_count,
            "top_application": None,
            "application_breakdown": [],
            "summary_text": "No desktop activity recorded.",
        }

    app_durations: Dict[str, float] = {}
    for s in sessions:
        app = s["application"]
        dur = s.get("duration_seconds", 0.0)
        app_durations[app] = app_durations.get(app, 0.0) + dur

    total_time = sum(app_durations.values())
    sorted_apps = sorted(app_durations.items(), key=lambda item: item[1], reverse=True)

    breakdown = []
    for app, secs in sorted_apps:
        pct = round((secs / total_time * 100), 1) if total_time > 0 else 0.0
        breakdown.append({
            "application": app,
            "seconds": secs,
            "formatted": format_duration_seconds(secs),
            "percentage": pct,
        })

    top_app = sorted_apps[0][0] if sorted_apps else None
    top_dur = format_duration_seconds(sorted_apps[0][1]) if sorted_apps else "0s"
    total_formatted = format_duration_seconds(total_time)

    # Formulate human-readable summary
    if len(sorted_apps) == 1:
        summary_text = (
            f"Observed {len(sessions)} session in {top_app} "
            f"totaling {total_formatted} (100% of recorded time)."
        )
    elif len(sorted_apps) == 2:
        second_app, second_secs = sorted_apps[1]
        summary_text = (
            f"Recorded {len(sessions)} sessions across 2 applications (total: {total_formatted}). "
            f"Primary focus was {top_app} ({top_dur}, {breakdown[0]['percentage']}%), "
            f"followed by {second_app} ({format_duration_seconds(second_secs)}, {breakdown[1]['percentage']}%)."
        )
    else:
        summary_text = (
            f"Recorded {len(sessions)} sessions across {len(sorted_apps)} applications (total: {total_formatted}). "
            f"Top application was {top_app} ({top_dur}, {breakdown[0]['percentage']}%)."
        )

    return {
        "total_time_seconds": total_time,
        "total_time_formatted": total_formatted,
        "total_sessions": len(sessions),
        "valid_events_count": valid_events_count,
        "invalid_events_count": invalid_events_count,
        "top_application": top_app,
        "application_breakdown": breakdown,
        "summary_text": summary_text,
    }


def generate_diagnostics(events: List[Any]) -> List[Dict[str, Any]]:
    """
    Run diagnostic audit on raw activity events.
    Flags invalid structures, empty OCR, duplicate timestamps, and temporal anomalies.
    """
    _, diagnostics = calculate_sessions(events)
    return diagnostics


def process_activity_events(events: List[Any]) -> Dict[str, Any]:
    """
    Complete end-to-end pipeline:
    Validates events, groups sessions, aggregates usage breakdown, and collects diagnostic warnings.
    """
    if not isinstance(events, list):
        return {
            "summary": {
                "total_time_seconds": 0.0,
                "total_time_formatted": "0s",
                "total_sessions": 0,
                "valid_events_count": 0,
                "invalid_events_count": 1,
                "top_application": None,
                "application_breakdown": [],
                "summary_text": "No valid event list provided.",
            },
            "sessions": [],
            "diagnostics": {
                "warning_count": 1,
                "warnings": [{
                    "severity": "warning",
                    "code": "INVALID_INPUT_TYPE",
                    "message": "Input events must be a list.",
                }],
            },
        }

    # Count valid vs invalid
    valid_count = 0
    invalid_count = 0
    for idx, ev in enumerate(events):
        ok, _, _ = validate_event(ev, index=idx)
        if ok:
            valid_count += 1
        else:
            invalid_count += 1

    sessions, diagnostics = calculate_sessions(events)
    summary = summarize_activity(
        sessions=sessions,
        valid_events_count=valid_count,
        invalid_events_count=invalid_count,
    )

    return {
        "summary": summary,
        "sessions": sessions,
        "diagnostics": {
            "warning_count": len(diagnostics),
            "warnings": diagnostics,
        },
    }


def get_demo_events() -> List[Dict[str, Any]]:
    """
    Provide realistic sample desktop events for verification and frontend prototype demo.
    Clearly labeled so mock events are never presented as live data.
    """
    now = time.time()
    return [
        {
            "application": "Visual Studio Code",
            "window_title": "ai.py — BLINDSPOT — Visual Studio Code",
            "activity": "coding",
            "timestamp": now - 3600,
            "end_time": now - 1800,
            "ocr_text": "class CompanionAI: def ask()",
            "confidence": 3,
        },
        {
            "application": "Google Chrome",
            "window_title": "FastAPI Documentation — Google Chrome",
            "activity": "researching in browser",
            "timestamp": now - 1800,
            "end_time": now - 900,
            "ocr_text": "FastAPI is a modern, fast web framework for building APIs",
            "confidence": 3,
        },
        {
            "application": "Slack",
            "window_title": "#engineering-standup — Slack",
            "activity": "chatting",
            "timestamp": now - 900,
            "end_time": now - 600,
            "ocr_text": "Standup notes: activity intelligence feature verified.",
            "confidence": 2,
        },
        {
            "application": "Visual Studio Code",
            "window_title": "activity_intelligence.py — BLINDSPOT",
            "activity": "coding",
            "timestamp": now - 600,
            "end_time": now,
            "ocr_text": "def process_activity_events(events)",
            "confidence": 3,
        },
        # One simulated transient empty OCR event for diagnostic demonstration
        {
            "application": "Google Chrome",
            "window_title": "Google Chrome",
            "activity": "browsing",
            "timestamp": now - 300,
            "ocr_text": "",
            "confidence": 1,
        },
    ]
