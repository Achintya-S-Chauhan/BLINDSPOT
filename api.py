"""
BLINDSPOT — Unified Activity Intelligence and Desktop Engine API

Exposes read-only and state-synchronization endpoints connecting the
running BLINDSPOT perception engine (Win32 window hooks, Tesseract OCR,
DesktopContext, ContextHistory, and Activity Intelligence) to the frontend dashboard.
"""

import os
import re
import time
import threading
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from activity_intelligence import (
    extract_app_name,
    format_duration_seconds,
    process_activity_events,
    get_demo_events,
    summarize_activity,
    calculate_sessions,
)

app = FastAPI(
    title="BLINDSPOT Engine & Activity Intelligence API",
    description="Real-time perception stream, DesktopContext, OCR buffers, history, and intelligence analytics.",
    version="2.0.0",
)

# Enable CORS for frontend dashboard (Vite default: 5173, preview: 4173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory engine state references
_engine_state: Optional[Dict[str, Any]] = None
_engine_lock: Optional[threading.Lock] = None
_engine_analyzer: Optional[Any] = None
_engine_stop_event: Optional[threading.Event] = None
_engine_thread: Optional[threading.Thread] = None
_engine_events_provider: Optional[Any] = None
_server_instance: Optional[Any] = None
_server_thread: Optional[threading.Thread] = None


def attach_engine_state(state: Dict[str, Any], lock: threading.Lock, analyzer: Optional[Any] = None):
    """Attach active BLINDSPOT runtime state and lock from main.py or ui.py."""
    global _engine_state, _engine_lock, _engine_analyzer
    _engine_state = state
    _engine_lock = lock
    _engine_analyzer = analyzer


def set_events_provider(provider):
    """Optional callback hook to supply custom event lists."""
    global _engine_events_provider
    _engine_events_provider = provider


def get_engine():
    """
    Get active engine state and lock. If not attached by an external runner,
    safely spin up the real BLINDSPOT perception engine in a background thread.
    """
    global _engine_state, _engine_lock, _engine_stop_event, _engine_thread, _engine_analyzer

    if _engine_state is not None and _engine_lock is not None:
        return _engine_state, _engine_lock, _engine_analyzer

    # Spin up standalone engine
    from main import initialize_system, monitor
    init_data = initialize_system(mock_io=False)
    _engine_state = init_data["state"]
    _engine_lock = init_data["state_lock"]
    _engine_analyzer = init_data["analyzer"]
    _engine_stop_event = init_data["stop_event"]

    _engine_thread = threading.Thread(
        target=monitor,
        args=(_engine_state, _engine_lock, _engine_stop_event),
        daemon=True,
    )
    _engine_thread.start()
    return _engine_state, _engine_lock, _engine_analyzer


def get_foreground_process_id() -> int:
    """Retrieve actual Win32 process ID of foreground window."""
    try:
        import win32gui
        import win32process
        hwnd = win32gui.GetForegroundWindow()
        if hwnd and win32gui.IsWindow(hwnd):
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            return int(pid)
    except Exception:
        pass
    return 0


def map_activity_to_category(activity: str) -> str:
    """Map BLINDSPOT interpreted activity string into standard dashboard category."""
    act = (activity or "").lower()
    if any(k in act for k in ("code", "coding", "dev", "program", "script", "terminal", "powershell", "cmd")):
        return "Development"
    if any(k in act for k in ("browser", "browsing", "read", "search", "google", "web")):
        return "Research"
    if any(k in act for k in ("chat", "slack", "discord", "teams", "message", "meeting")):
        return "Communication"
    if any(k in act for k in ("notion", "note", "word", "write", "writing", "pdf", "doc")):
        return "Documentation"
    if any(k in act for k in ("video", "youtube", "media", "music", "spotify", "watch")):
        return "Media"
    if any(k in act for k in ("figma", "design", "paint", "canvas")):
        return "Design"
    return "Productivity"


def extract_keywords(ocr_text: str, max_keywords: int = 6) -> List[str]:
    """Deterministically extract high-frequency descriptive words from OCR buffer."""
    if not ocr_text:
        return []
    words = re.findall(r"[A-Za-z0-9_]{4,}", ocr_text)
    stop = {
        "this", "that", "with", "from", "have", "were", "what", "when",
        "your", "more", "file", "line", "true", "false", "none", "null",
    }
    counts: Dict[str, int] = {}
    for w in words:
        wl = w.lower()
        if wl not in stop and not wl.isdigit():
            counts[w] = counts.get(w, 0) + 1
    sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [w[0] for w in sorted_words[:max_keywords]]


def get_real_engine_events() -> List[Dict[str, Any]]:
    """
    Extract genuine completed and active context events from running engine state.
    Returns empty list if engine has not yet captured events.
    """
    state, lock, _ = get_engine()
    now = time.time()
    events = []

    with lock:
        # 1. Stored completed history records
        if state.get("history") is not None:
            records = state["history"].get_recent(limit=50)
            for r in records:
                events.append({
                    "application": extract_app_name(r.window_title),
                    "window_title": r.window_title,
                    "activity": r.activity,
                    "timestamp": r.start_time,
                    "end_time": r.end_time,
                    "duration": r.duration,
                    "ocr_text": getattr(r.context, "ocr_text", "") if hasattr(r, "context") else "",
                    "confidence": r.confidence,
                })

        # 2. Ongoing active context episode
        cur_ctx = state.get("current_context")
        start_time = state.get("context_start_time")
        if cur_ctx is not None and start_time is not None:
            events.append({
                "application": extract_app_name(cur_ctx.window_title),
                "window_title": cur_ctx.window_title,
                "activity": cur_ctx.activity,
                "timestamp": start_time,
                "end_time": now,
                "duration": max(0.0, now - start_time),
                "ocr_text": cur_ctx.ocr_text,
                "confidence": cur_ctx.confidence,
            })

    return events


# ---------------- API ENDPOINTS ----------------

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "BLINDSPOT Engine & Activity Intelligence API",
        "version": "2.0.0",
        "endpoints": [
            "/api/health",
            "/api/engine/status",
            "/api/context/live",
            "/api/context/screen",
            "/api/activity/history",
            "/api/overview",
            "/api/activity/intelligence",
            "/api/settings",
            "/api/activity/demo",
            "/api/activity/diagnose",
        ],
    }


@app.get("/api/health")
def get_health():
    """Verify backend and perception engine connection health."""
    state, lock, _ = get_engine()
    with lock:
        is_running = state.get("is_running", False)
        backend_io = state.get("desktop_io")
        backend_name = backend_io.__class__.__name__ if backend_io else "Unknown"

    return {
        "status": "online",
        "engine_running": is_running,
        "is_demo": False,
        "backend_io": backend_name,
        "timestamp": time.time(),
    }


@app.get("/api/engine/status")
def get_engine_status():
    """Detailed perception loop diagnostics and running state."""
    state, lock, _ = get_engine()
    with lock:
        return {
            "status": "online",
            "is_running": state.get("is_running", True),
            "last_window": state.get("last_window"),
            "last_context": state.get("last_context"),
            "last_confidence": state.get("last_confidence", 0),
            "last_ocr_time": state.get("last_ocr_time"),
            "context_start_time": state.get("context_start_time"),
            "history_count": len(state.get("history", [])),
            "activity_time_keys": list(state.get("activity_time", {}).keys()),
        }


@app.post("/api/engine/toggle")
def toggle_engine(body: Optional[Dict[str, Any]] = Body(None)):
    """Pause or resume the background desktop perception loop."""
    state, lock, _ = get_engine()
    with lock:
        if body and "running" in body:
            new_running = bool(body["running"])
        else:
            new_running = not state.get("is_running", True)
        state["is_running"] = new_running

    return {
        "status": "success",
        "is_running": new_running,
    }


@app.get("/api/context/live")
def get_live_context():
    """
    Return genuine live active foreground window, DesktopContext snapshot,
    focus duration, bounding rectangle, and recent application switch history.
    """
    state, lock, analyzer = get_engine()
    now = time.time()
    pid = get_foreground_process_id()

    with lock:
        is_running = state.get("is_running", True)
        cur_ctx = state.get("current_context")
        last_win = state.get("last_window") or ""
        last_ctx = state.get("last_context") or "idle"
        confidence = state.get("last_confidence", 0)
        start_time = state.get("context_start_time")
        last_ocr = state.get("last_ocr_time")
        history = state.get("history")

        # Duration
        dur = max(0.0, now - start_time) if start_time else 0.0

        # Window bounds
        bounds = {"x": 0, "y": 0, "width": 1920, "height": 1080}
        if cur_ctx and cur_ctx.observation.window_rect:
            rect = cur_ctx.observation.window_rect
            bounds = {
                "x": rect.get("left", 0),
                "y": rect.get("top", 0),
                "width": rect.get("width", 1920),
                "height": rect.get("height", 1080),
            }

        # Analyze workflow pattern if analyzer available
        pattern = "focused_session"
        interpretation = f"Active focus in {extract_app_name(last_win)}"
        if analyzer is not None:
            try:
                und = analyzer.analyze(
                    current_context=cur_ctx,
                    context_start_time=start_time,
                    history=history,
                    now=now,
                )
                pattern = und.pattern_type
                interpretation = und.interpretation or interpretation
            except Exception:
                pass

        # Build genuine switch history from ContextHistory
        recent_switches = []
        if history is not None:
            records = history.get_recent(limit=10)
            for i in range(1, len(records)):
                prev_app = extract_app_name(records[i - 1].window_title)
                curr_app = extract_app_name(records[i].window_title)
                if prev_app != curr_app:
                    recent_switches.insert(0, {
                        "id": f"sw-{int(records[i].start_time * 1000)}",
                        "fromApp": prev_app,
                        "toApp": curr_app,
                        "timestamp": time.strftime("%H:%M:%S", time.localtime(records[i].start_time)),
                        "durationInApp": format_duration_seconds(records[i - 1].duration),
                    })

    app_name = extract_app_name(last_win)
    category = map_activity_to_category(last_ctx)

    return {
        "status": "success",
        "is_demo": False,
        "data_source": "active_engine",
        "context": {
            "activeApplication": app_name,
            "activeWindow": last_win,
            "category": category,
            "activity": last_ctx,
            "confidence": confidence,
            "processId": pid,
            "sessionStartTime": (start_time * 1000) if start_time else (now * 1000),
            "currentDurationSeconds": int(dur),
            "currentDurationFormatted": format_duration_seconds(dur),
            "lastOcrTime": (last_ocr * 1000) if last_ocr else None,
            "isMonitoring": is_running,
            "windowBounds": bounds,
            "workflowPattern": pattern,
            "workflowInterpretation": interpretation,
            "recentSwitches": recent_switches[:8],
        },
    }


@app.get("/api/context/screen")
def get_screen_context():
    """
    Return genuine DesktopContext and raw in-memory OCR text buffer
    perceived by Tesseract for the foreground window.
    """
    state, lock, _ = get_engine()
    with lock:
        cur_ctx = state.get("current_context")
        last_win = state.get("last_window") or ""
        last_ocr = state.get("last_ocr_time")

        if cur_ctx is not None:
            win_title = cur_ctx.window_title
            raw_text = cur_ctx.ocr_text or ""
            confidence = cur_ctx.confidence
            activity = cur_ctx.activity
            rect = cur_ctx.observation.window_rect or {"left": 0, "top": 0, "width": 1920, "height": 1080}
        else:
            win_title = last_win
            raw_text = ""
            confidence = 0
            activity = "no_context"
            rect = {"left": 0, "top": 0, "width": 1920, "height": 1080}

    words = raw_text.split()
    keywords = extract_keywords(raw_text)
    captured_str = (
        time.strftime("%H:%M:%S", time.localtime(last_ocr))
        if last_ocr
        else "No OCR frames captured yet"
    )

    return {
        "status": "success",
        "is_demo": False,
        "data_source": "active_engine",
        "context": {
            "windowTitle": win_title or "No active foreground window",
            "rawOcrText": raw_text,
            "wordCount": len(words),
            "characterCount": len(raw_text),
            "detectedKeywords": keywords,
            "capturedAt": captured_str,
            "confidence": confidence,
            "activity": activity,
            "windowRect": rect,
        },
    }


@app.get("/api/activity/history")
def get_activity_history():
    """
    Return genuine historical episodes from ContextHistory and the ongoing active context.
    Never mixes simulated records into live responses.
    """
    state, lock, _ = get_engine()
    now = time.time()
    records_out = []
    total_seconds = 0.0
    app_durations: Dict[str, float] = {}

    with lock:
        history = state.get("history")
        if history is not None:
            records = history.get_recent(limit=50)
            for r in records:
                dur = r.duration
                total_seconds += dur
                app_name = extract_app_name(r.window_title)
                app_durations[app_name] = app_durations.get(app_name, 0.0) + dur

                ocr_snippet = ""
                if hasattr(r, "context") and hasattr(r.context, "ocr_text"):
                    ocr_snippet = (r.context.ocr_text or "").strip()[:140]

                records_out.append({
                    "id": f"rec-{int(r.start_time * 1000)}",
                    "application": app_name,
                    "windowTitle": r.window_title,
                    "category": map_activity_to_category(r.activity),
                    "activity": r.activity,
                    "startTime": time.strftime("%H:%M:%S", time.localtime(r.start_time)),
                    "endTime": time.strftime("%H:%M:%S", time.localtime(r.end_time)) if r.end_time else "Active",
                    "durationSeconds": int(dur),
                    "durationFormatted": format_duration_seconds(dur),
                    "confidence": r.confidence,
                    "date": "Today",
                    "ocrSnippet": ocr_snippet,
                })

        # Include current active episode
        cur_ctx = state.get("current_context")
        start_time = state.get("context_start_time")
        if cur_ctx is not None and start_time is not None:
            active_dur = max(0.0, now - start_time)
            total_seconds += active_dur
            app_name = extract_app_name(cur_ctx.window_title)
            app_durations[app_name] = app_durations.get(app_name, 0.0) + active_dur

            records_out.insert(0, {
                "id": f"rec-active-{int(start_time * 1000)}",
                "application": app_name,
                "windowTitle": cur_ctx.window_title,
                "category": map_activity_to_category(cur_ctx.activity),
                "activity": cur_ctx.activity,
                "startTime": time.strftime("%H:%M:%S", time.localtime(start_time)),
                "endTime": "Active",
                "durationSeconds": int(active_dur),
                "durationFormatted": format_duration_seconds(active_dur),
                "confidence": cur_ctx.confidence,
                "date": "Today",
                "ocrSnippet": (cur_ctx.ocr_text or "").strip()[:140],
            })

    top_app = "None"
    if app_durations:
        top_app = max(app_durations.items(), key=lambda x: x[1])[0]

    avg_dur = (total_seconds / len(records_out)) if records_out else 0.0

    return {
        "status": "success",
        "is_demo": False,
        "data_source": "active_engine",
        "summary": {
            "totalRecords": len(records_out),
            "totalTimeSeconds": int(total_seconds),
            "totalTimeFormatted": format_duration_seconds(total_seconds),
            "topApplication": top_app,
            "deepWorkAvgFormatted": format_duration_seconds(avg_dur),
        },
        "records": records_out,
    }


@app.get("/api/overview")
def get_overview_metrics():
    """
    Return genuine aggregated overview statistics, application time breakdown,
    category totals, and recent timeline based entirely on real engine state.
    """
    state, lock, analyzer = get_engine()
    now = time.time()

    with lock:
        is_running = state.get("is_running", True)
        cur_ctx = state.get("current_context")
        last_win = state.get("last_window") or ""
        last_ctx = state.get("last_context") or "idle"
        start_time = state.get("context_start_time")
        activity_times = dict(state.get("activity_time", {}))
        history = state.get("history")

        # Current active duration
        active_dur = max(0.0, now - start_time) if start_time else 0.0
        if last_ctx:
            activity_times[last_ctx] = activity_times.get(last_ctx, 0.0) + active_dur

        # App usage calculations from history + active
        app_seconds_map: Dict[str, float] = {}
        observed_apps = set()
        if history is not None:
            for r in history.get_recent(limit=50):
                aname = extract_app_name(r.window_title)
                observed_apps.add(aname)
                app_seconds_map[aname] = app_seconds_map.get(aname, 0.0) + r.duration

        cur_app = extract_app_name(last_win)
        if cur_app != "Unknown":
            observed_apps.add(cur_app)
            app_seconds_map[cur_app] = app_seconds_map.get(cur_app, 0.0) + active_dur

        # Category totals
        cat_totals: Dict[str, Dict[str, Any]] = {}
        for act, sec in activity_times.items():
            cat = map_activity_to_category(act)
            if cat not in cat_totals:
                colors = {
                    "Development": "#6366f1",
                    "Research": "#3b82f6",
                    "Communication": "#10b981",
                    "Documentation": "#f59e0b",
                    "Media": "#8b5cf6",
                    "Design": "#ec4899",
                    "Productivity": "#64748b",
                }
                cat_totals[cat] = {
                    "seconds": 0.0,
                    "color": colors.get(cat, "#64748b"),
                    "count": 0,
                }
            cat_totals[cat]["seconds"] += sec
            cat_totals[cat]["count"] += 1

        total_tracked = sum(activity_times.values())

        # Top apps list
        app_usage_list = []
        for aname, asecs in sorted(app_seconds_map.items(), key=lambda x: x[1], reverse=True):
            pct = round((asecs / total_tracked * 100), 1) if total_tracked > 0 else 0.0
            app_usage_list.append({
                "name": aname,
                "category": map_activity_to_category(aname),
                "totalSeconds": int(asecs),
                "formattedDuration": format_duration_seconds(asecs),
                "percentage": pct,
                "color": "#6366f1",
            })

        # Recent timeline nodes (last 5)
        timeline_nodes = []
        if history is not None:
            for r in history.get_recent(limit=5):
                timeline_nodes.append({
                    "id": f"tl-{int(r.start_time * 1000)}",
                    "application": extract_app_name(r.window_title),
                    "category": map_activity_to_category(r.activity),
                    "startTime": time.strftime("%H:%M:%S", time.localtime(r.start_time)),
                    "endTime": time.strftime("%H:%M:%S", time.localtime(r.end_time)) if r.end_time else "Active",
                    "durationFormatted": format_duration_seconds(r.duration),
                    "ocrSnippet": getattr(r.context, "ocr_text", "")[:100] if hasattr(r, "context") else "",
                })

        # Pattern
        pattern = "focused_session"
        if analyzer is not None:
            try:
                und = analyzer.analyze(
                    current_context=cur_ctx,
                    context_start_time=start_time,
                    history=history,
                    now=now,
                )
                pattern = und.pattern_type
            except Exception:
                pass

    pid = get_foreground_process_id()

    return {
        "status": "success",
        "is_demo": False,
        "data_source": "active_engine",
        "overview": {
            "activeApplication": cur_app,
            "processId": pid,
            "isMonitoring": is_running,
            "trackedTimeTodaySeconds": int(total_tracked),
            "trackedTimeTodayFormatted": format_duration_seconds(total_tracked),
            "applicationsObservedCount": len(observed_apps),
            "workflowPattern": pattern,
            "categoryTotals": cat_totals,
            "appUsage": app_usage_list[:5],
            "recentTimeline": timeline_nodes,
        },
    }


@app.get("/api/activity/intelligence")
def get_activity_intelligence(
    use_demo: bool = Query(False, description="Explicitly request demonstration dataset instead of real engine events"),
):
    """
    Process desktop events through Activity Intelligence pipeline.
    When use_demo is False, processes actual events from the running BLINDSPOT engine.
    """
    is_demo = False
    events = []

    if _engine_events_provider is not None:
        try:
            events = _engine_events_provider()
        except Exception:
            events = []

    # If no provider injected, collect actual engine events from state
    if not events and not use_demo:
        events = get_real_engine_events()

    # If user explicitly asked for demo or engine has 0 events and demo requested
    if use_demo:
        events = get_demo_events()
        is_demo = True

    processed = process_activity_events(events)
    return {
        "status": "success",
        "is_demo": is_demo,
        "data_source": "demo_dataset" if is_demo else "active_engine",
        "intelligence": processed,
    }


@app.get("/api/activity/demo")
def get_demo_stream():
    """Return raw demonstration event dataset."""
    return {
        "status": "success",
        "is_demo": True,
        "events": get_demo_events(),
    }


@app.post("/api/activity/diagnose")
def diagnose_events(events: List[Dict[str, Any]] = Body(...)):
    """Accept arbitrary activity events and run validation & diagnostics."""
    if not isinstance(events, list):
        raise HTTPException(status_code=400, detail="Request body must be a list of activity events.")

    result = process_activity_events(events)
    return {
        "status": "success",
        "is_demo": False,
        "diagnosed_event_count": len(events),
        "intelligence": result,
    }


@app.get("/api/settings")
def get_settings():
    """
    Return true runtime configuration parameters enforced by the BLINDSPOT engine.
    Only returns parameters actually implemented by the backend.
    """
    state, lock, _ = get_engine()
    with lock:
        backend_io = state.get("desktop_io")
        backend_name = backend_io.__class__.__name__ if backend_io else "WindowsDesktopIO"

    return {
        "pollIntervalMs": 300,
        "ocrIntervalSeconds": 3.0,
        "stabilityTimeSeconds": 1.5,
        "minContextDurationSeconds": 3.0,
        "historyMaxlen": 50,
        "inMemoryOnly": True,
        "backendIo": backend_name,
        "privacyMode": "Local-Only in RAM (Zero Screengrabs Saved)",
    }


# Background thread server utility
def start_api_server_background(host="127.0.0.1", port=8000):
    """Start uvicorn in a daemon thread so it runs alongside CLI/Tkinter."""
    global _server_thread, _server_instance
    if _server_thread is not None and _server_thread.is_alive():
        return
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    _server_instance = uvicorn.Server(config)
    _server_thread = threading.Thread(target=_server_instance.run, daemon=True)
    _server_thread.start()


if __name__ == "__main__":
    import uvicorn
    # Standalone execution: ensures engine is initialized and running
    get_engine()
    print("[BLINDSPOT API] Active perception engine online. Serving API at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
