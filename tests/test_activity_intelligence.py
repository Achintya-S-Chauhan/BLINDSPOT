"""
Unit and integration tests for BLINDSPOT Activity Intelligence and Diagnostics.

Tests:
1. Event validation (valid, missing fields, malformed timestamps, non-dict input)
2. Session calculation (app grouping, boundary transitions, duration logic)
3. Duplicate event detection and diagnostics (SUSPICIOUS_DUPLICATE)
4. Empty OCR buffer warnings (EMPTY_OCR)
5. Out-of-order timestamp handling (OUT_OF_ORDER_TIMESTAMPS)
6. Activity summaries (aggregates, percentages, human-readable summary text)
7. Empty and degenerate inputs
8. Real DesktopContext and ContextRecord interoperability
9. FastAPI endpoints integration
"""

import time
import pytest
from fastapi.testclient import TestClient

from activity_intelligence import (
    extract_app_name,
    format_duration_seconds,
    normalize_event,
    validate_event,
    calculate_sessions,
    summarize_activity,
    generate_diagnostics,
    process_activity_events,
    get_demo_events,
)
from context import DesktopContext, RawObservation, InterpretedContext
from history import ContextRecord
from api import app, set_events_provider


# ----------------------------------------------------------------------
# 1. Helper function tests: format_duration_seconds & extract_app_name
# ----------------------------------------------------------------------

def test_format_duration_seconds():
    assert format_duration_seconds(0) == "0s"
    assert format_duration_seconds(-10) == "0s"
    assert format_duration_seconds(45) == "45s"
    assert format_duration_seconds(60) == "1m"
    assert format_duration_seconds(125) == "2m 5s"
    assert format_duration_seconds(3600) == "1h 0m"
    assert format_duration_seconds(3750) == "1h 2m"


def test_extract_app_name():
    # Known canonical desktop apps
    assert extract_app_name("main.py — BLINDSPOT — Visual Studio Code") == "Visual Studio Code"
    assert extract_app_name("GitHub - Pull Requests - Google Chrome") == "Google Chrome"
    assert extract_app_name("Untitled - Notepad") == "Notepad"
    assert extract_app_name("Calculator") == "Calculator"
    assert extract_app_name("Windows Terminal") == "Windows Terminal"
    assert extract_app_name("Slack - #general") == "Slack"

    # Segment fallback
    assert extract_app_name("MyDocument.docx - WordPad") == "WordPad"

    # Degenerate cases
    assert extract_app_name("") == "Unknown"
    assert extract_app_name("   ") == "Unknown"
    assert extract_app_name(None) == "Unknown"
    assert extract_app_name("No active window detected") == "Unknown"


# ----------------------------------------------------------------------
# 2. Event Validation Tests
# ----------------------------------------------------------------------

def test_validate_event_valid():
    ev = {
        "application": "Visual Studio Code",
        "window_title": "test.py — VS Code",
        "timestamp": 1700000000.0,
        "activity": "coding",
        "ocr_text": "def test_func(): pass",
    }
    is_valid, sanitized, warnings = validate_event(ev)
    assert is_valid is True
    assert sanitized is not None
    assert sanitized["application"] == "Visual Studio Code"
    assert sanitized["timestamp"] == 1700000000.0
    assert sanitized["activity"] == "coding"
    assert len([w for w in warnings if w["severity"] == "warning"]) == 0


def test_validate_event_missing_identifier():
    ev = {
        "timestamp": 1700000000.0,
        "activity": "coding",
    }
    is_valid, sanitized, warnings = validate_event(ev, index=0)
    assert is_valid is False
    assert sanitized is None
    codes = [w["code"] for w in warnings]
    assert "MISSING_IDENTIFIER" in codes


def test_validate_event_missing_timestamp():
    ev = {
        "application": "Spotify",
        "window_title": "Spotify Free",
    }
    is_valid, sanitized, warnings = validate_event(ev, index=1)
    assert is_valid is False
    assert sanitized is None
    codes = [w["code"] for w in warnings]
    assert "MISSING_TIMESTAMP" in codes


def test_validate_event_malformed_and_negative_timestamp():
    # Negative timestamp
    ev_neg = {
        "application": "Notepad",
        "timestamp": -5.0,
    }
    is_valid, sanitized, warnings = validate_event(ev_neg, index=2)
    assert is_valid is False
    assert "INVALID_TIMESTAMP" in [w["code"] for w in warnings]

    # Non-numeric timestamp string
    ev_str = {
        "application": "Notepad",
        "timestamp": "invalid_date_str",
    }
    is_valid, sanitized, warnings = validate_event(ev_str, index=3)
    assert is_valid is False
    assert "MALFORMED_TIMESTAMP" in [w["code"] for w in warnings]


def test_validate_event_invalid_type():
    is_valid, sanitized, warnings = validate_event(None, index=0)
    assert is_valid is False
    assert "INVALID_EVENT" in [w["code"] for w in warnings]


def test_validate_event_empty_ocr_diagnostic():
    # Text-heavy app with empty OCR should raise an EMPTY_OCR notice
    ev = {
        "application": "Visual Studio Code",
        "window_title": "empty.py - Visual Studio Code",
        "timestamp": 1700000000.0,
        "ocr_text": "   ",
    }
    is_valid, sanitized, warnings = validate_event(ev)
    assert is_valid is True
    assert "EMPTY_OCR" in [w["code"] for w in warnings]

    # Non text-heavy app should not trigger warning
    ev_spotify = {
        "application": "Spotify",
        "window_title": "Spotify",
        "timestamp": 1700000000.0,
        "ocr_text": "",
    }
    _, _, warnings_spotify = validate_event(ev_spotify)
    assert "EMPTY_OCR" not in [w["code"] for w in warnings_spotify]


# ----------------------------------------------------------------------
# 3. Session Calculation Tests
# ----------------------------------------------------------------------

def test_calculate_sessions_grouping_and_boundaries():
    t0 = 1000.0
    events = [
        {"application": "Visual Studio Code", "window_title": "a.py", "timestamp": t0},
        {"application": "Visual Studio Code", "window_title": "b.py", "timestamp": t0 + 60},
        {"application": "Google Chrome", "window_title": "Search", "timestamp": t0 + 180},
        {"application": "Visual Studio Code", "window_title": "a.py", "timestamp": t0 + 300},
    ]

    sessions, diagnostics = calculate_sessions(events)
    # Consecutive VS Code events merged:
    # Session 1: VS Code (t0 to t0+60 -> closed at next event t0+180)
    # Session 2: Chrome (t0+180 -> closed at t0+300)
    # Session 3: VS Code (t0+300 to t0+300 + min_duration)
    assert len(sessions) == 3
    assert sessions[0]["application"] == "Visual Studio Code"
    assert sessions[0]["event_count"] == 2
    assert "a.py" in sessions[0]["window_titles"]
    assert "b.py" in sessions[0]["window_titles"]
    assert sessions[0]["duration_seconds"] == 180.0

    assert sessions[1]["application"] == "Google Chrome"
    assert sessions[1]["event_count"] == 1
    assert sessions[1]["duration_seconds"] == 120.0

    assert sessions[2]["application"] == "Visual Studio Code"
    assert sessions[2]["event_count"] == 1


def test_calculate_sessions_duration_with_explicit_end_time():
    t0 = 1000.0
    events = [
        {
            "application": "Figma",
            "window_title": "Design Mockup",
            "timestamp": t0,
            "end_time": t0 + 600,
        }
    ]
    sessions, _ = calculate_sessions(events)
    assert len(sessions) == 1
    assert sessions[0]["duration_seconds"] == 600.0
    assert sessions[0]["duration_formatted"] == "10m"


def test_calculate_sessions_duplicate_detection():
    t0 = 1000.0
    events = [
        {"application": "Notepad", "window_title": "Notes.txt", "timestamp": t0},
        {"application": "Notepad", "window_title": "Notes.txt", "timestamp": t0},  # Duplicate
        {"application": "Notepad", "window_title": "Notes.txt", "timestamp": t0 + 30},
    ]

    sessions, diagnostics = calculate_sessions(events)
    assert len(sessions) == 1
    # Duplicate was filtered from sessions and recorded in diagnostics
    dup_warnings = [d for d in diagnostics if d["code"] == "SUSPICIOUS_DUPLICATE"]
    assert len(dup_warnings) == 1
    assert sessions[0]["event_count"] == 2


def test_calculate_sessions_out_of_order_timestamps():
    t0 = 1000.0
    events = [
        {"application": "Slack", "window_title": "#dev", "timestamp": t0 + 200},
        {"application": "Slack", "window_title": "#dev", "timestamp": t0},  # retrograde
    ]

    sessions, diagnostics = calculate_sessions(events)
    # Chronological sort performed, diagnostic warning recorded
    out_order_warns = [d for d in diagnostics if d["code"] == "OUT_OF_ORDER_TIMESTAMPS"]
    assert len(out_order_warns) == 1
    assert len(sessions) == 1
    assert sessions[0]["start_time"] == t0


def test_calculate_sessions_empty_and_invalid():
    assert calculate_sessions([]) == ([], [])
    assert calculate_sessions(None) == ([], [])

    # List containing only invalid events
    bad_events = [{"invalid": "data"}, None]
    sessions, diagnostics = calculate_sessions(bad_events)
    assert sessions == []
    assert len(diagnostics) == 2


# ----------------------------------------------------------------------
# 4. Activity Summaries Tests
# ----------------------------------------------------------------------

def test_summarize_activity_empty():
    summary = summarize_activity([], valid_events_count=0, invalid_events_count=0)
    assert summary["total_time_seconds"] == 0.0
    assert summary["total_sessions"] == 0
    assert summary["top_application"] is None
    assert summary["summary_text"] == "No desktop activity recorded."


def test_summarize_activity_single_app():
    sessions = [
        {
            "application": "Visual Studio Code",
            "duration_seconds": 1800.0,
            "duration_formatted": "30m",
            "event_count": 5,
        }
    ]
    summary = summarize_activity(sessions, valid_events_count=5, invalid_events_count=0)
    assert summary["total_time_seconds"] == 1800.0
    assert summary["total_time_formatted"] == "30m"
    assert summary["top_application"] == "Visual Studio Code"
    assert len(summary["application_breakdown"]) == 1
    assert summary["application_breakdown"][0]["percentage"] == 100.0
    assert "Observed 1 session in Visual Studio Code totaling 30m" in summary["summary_text"]


def test_summarize_activity_multiple_apps():
    sessions = [
        {"application": "Visual Studio Code", "duration_seconds": 3600.0},
        {"application": "Google Chrome", "duration_seconds": 1800.0},
        {"application": "Slack", "duration_seconds": 600.0},
    ]
    summary = summarize_activity(sessions, valid_events_count=10, invalid_events_count=1)
    assert summary["total_time_seconds"] == 6000.0
    assert summary["total_time_formatted"] == "1h 40m"
    assert summary["top_application"] == "Visual Studio Code"
    assert len(summary["application_breakdown"]) == 3
    # Percentages check: VS Code (60%), Chrome (30%), Slack (10%)
    breakdown_map = {b["application"]: b["percentage"] for b in summary["application_breakdown"]}
    assert breakdown_map["Visual Studio Code"] == 60.0
    assert breakdown_map["Google Chrome"] == 30.0
    assert breakdown_map["Slack"] == 10.0
    assert "Top application was Visual Studio Code (1h 0m, 60.0%)" in summary["summary_text"]


# ----------------------------------------------------------------------
# 5. Full Pipeline & Demo Events Tests
# ----------------------------------------------------------------------

def test_process_activity_events_pipeline():
    demo = get_demo_events()
    assert isinstance(demo, list)
    assert len(demo) >= 4

    result = process_activity_events(demo)
    assert "summary" in result
    assert "sessions" in result
    assert "diagnostics" in result
    assert result["summary"]["total_sessions"] > 0
    assert result["summary"]["valid_events_count"] == len(demo)
    assert result["diagnostics"]["warning_count"] >= 1  # includes the transient empty OCR event


def test_process_activity_events_non_list_input():
    result = process_activity_events("not a list")
    assert result["summary"]["total_sessions"] == 0
    assert result["diagnostics"]["warning_count"] == 1
    assert result["diagnostics"]["warnings"][0]["code"] == "INVALID_INPUT_TYPE"


# ----------------------------------------------------------------------
# 6. DesktopContext and ContextRecord Interoperability Tests
# ----------------------------------------------------------------------

def test_desktop_context_interoperability():
    obs = RawObservation(
        window_title="main.py - BLINDSPOT - Visual Studio Code",
        ocr_text="import os",
        timestamp=1000.0,
    )
    interp = InterpretedContext(activity="coding", confidence=3)
    ctx = DesktopContext(observation=obs, interpretation=interp, timestamp=1000.0)

    # Validate normalize_event
    norm = normalize_event(ctx)
    assert norm["application"] == "Visual Studio Code"
    assert norm["window_title"] == "main.py - BLINDSPOT - Visual Studio Code"
    assert norm["activity"] == "coding"
    assert norm["timestamp"] == 1000.0

    # Pass DesktopContext objects to process_activity_events
    obs2 = RawObservation(
        window_title="FastAPI Documentation - Google Chrome",
        ocr_text="FastAPI intro",
        timestamp=1500.0,
    )
    ctx2 = DesktopContext(observation=obs2, interpretation=interp, timestamp=1500.0)

    result = process_activity_events([ctx, ctx2])
    assert result["summary"]["total_sessions"] == 2
    assert result["summary"]["valid_events_count"] == 2


def test_context_record_interoperability():
    obs = RawObservation(
        window_title="Untitled - Notepad",
        ocr_text="Todo list",
        timestamp=2000.0,
    )
    ctx = DesktopContext(
        observation=obs,
        interpretation=InterpretedContext(activity="writing notes", confidence=2),
    )
    record = ContextRecord(
        context=ctx,
        start_time=2000.0,
        end_time=2300.0,
        duration=300.0,
    )

    norm = normalize_event(record)
    assert norm["application"] == "Notepad"
    assert norm["timestamp"] == 2000.0
    assert norm["end_time"] == 2300.0

    result = process_activity_events([record])
    assert result["summary"]["total_sessions"] == 1
    assert result["sessions"][0]["duration_seconds"] == 300.0


# ----------------------------------------------------------------------
# 7. FastAPI Endpoints Integration Tests
# ----------------------------------------------------------------------

client = TestClient(app)


def test_api_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/api/activity/intelligence" in data["endpoints"]


def test_api_activity_intelligence_with_demo():
    # Reset provider to ensure demo dataset path is tested
    set_events_provider(None)
    response = client.get("/api/activity/intelligence?use_demo=true")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_demo"] is True
    assert data["data_source"] == "demo_dataset"
    assert "intelligence" in data
    assert data["intelligence"]["summary"]["total_sessions"] > 0


def test_api_activity_intelligence_no_demo_empty():
    set_events_provider(None)
    response = client.get("/api/activity/intelligence?use_demo=false")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_demo"] is False
    assert data["data_source"] == "active_engine"
    assert data["intelligence"]["summary"]["total_sessions"] == 0


def test_api_demo_stream_endpoint():
    response = client.get("/api/activity/demo")
    assert response.status_code == 200
    data = response.json()
    assert data["is_demo"] is True
    assert len(data["events"]) > 0


def test_api_diagnose_post_endpoint():
    payload = [
        {
            "application": "Calculator",
            "window_title": "Calculator",
            "timestamp": 5000.0,
            "end_time": 5060.0,
        },
        {
            "application": "Visual Studio Code",
            "window_title": "api.py",
            "timestamp": 5060.0,
            "ocr_text": "",  # triggers EMPTY_OCR info warning
        },
    ]
    response = client.post("/api/activity/diagnose", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["diagnosed_event_count"] == 2
    assert data["intelligence"]["summary"]["total_sessions"] == 2
    assert data["intelligence"]["diagnostics"]["warning_count"] >= 1


def test_api_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "engine_running" in data
    assert data["is_demo"] is False


def test_api_engine_status_and_toggle():
    status_res = client.get("/api/engine/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["status"] == "online"
    assert "is_running" in status_data

    # Toggle running state
    toggle_res = client.post("/api/engine/toggle", json={"running": False})
    assert toggle_res.status_code == 200
    assert toggle_res.json()["is_running"] is False

    # Restore running state
    toggle_res2 = client.post("/api/engine/toggle", json={"running": True})
    assert toggle_res2.status_code == 200
    assert toggle_res2.json()["is_running"] is True


def test_api_live_context_endpoint():
    response = client.get("/api/context/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_demo"] is False
    assert "context" in data
    ctx = data["context"]
    assert "activeApplication" in ctx
    assert "activeWindow" in ctx
    assert "currentDurationSeconds" in ctx
    assert "recentSwitches" in ctx
    assert "workflowPattern" in ctx


def test_api_screen_context_endpoint():
    response = client.get("/api/context/screen")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_demo"] is False
    assert "context" in data
    ctx = data["context"]
    assert "windowTitle" in ctx
    assert "rawOcrText" in ctx
    assert "wordCount" in ctx
    assert "characterCount" in ctx
    assert "detectedKeywords" in ctx


def test_api_activity_history_endpoint():
    response = client.get("/api/activity/history")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_demo"] is False
    assert "summary" in data
    assert "records" in data
    assert isinstance(data["records"], list)


def test_api_overview_endpoint():
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_demo"] is False
    assert "overview" in data
    ov = data["overview"]
    assert "activeApplication" in ov
    assert "trackedTimeTodaySeconds" in ov
    assert "categoryTotals" in ov
    assert "appUsage" in ov
    assert "recentTimeline" in ov


def test_api_settings_endpoint():
    response = client.get("/api/settings")
    assert response.status_code == 200
    data = response.json()
    assert data["pollIntervalMs"] == 300
    assert data["ocrIntervalSeconds"] == 3.0
    assert data["inMemoryOnly"] is True
    assert "backendIo" in data
