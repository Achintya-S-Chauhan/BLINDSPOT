import time
import threading
from eyes.window import get_active_window, get_active_window_rect
from eyes.screen import capture_screen
from eyes.ocr import read_text_from_image
from eyes.interpreter import interpret_context
from context import RawObservation, InterpretedContext, DesktopContext
from history import ContextHistory
from understanding import ContextualUnderstandingAnalyzer

STABILITY_TIME = 1.5        # Time window must remain active before first OCR
OCR_INTERVAL = 3.0          # Interval between periodic OCR scans in the same active window
MIN_CONTEXT_DURATION = 3.0  # Minimum candidate context duration before switching
POLL_INTERVAL = 0.3         # Background poll loop sleep


def format_time(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    hours = minutes // 60

    if hours > 0:
        return f"{hours}h {minutes % 60}m"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return f"{seconds}s"


def format_timestamp(timestamp):
    """Format Unix timestamp into HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.localtime(timestamp))


def close_active_context(state, now):
    """Close the ongoing context, update cumulative activity time, and record into history."""
    if state["last_context"] and state["context_start_time"]:
        duration = now - state["context_start_time"]
        state["activity_time"][state["last_context"]] = (
            state["activity_time"].get(state["last_context"], 0) + duration
        )
        ctx_obj = state.get("previous_context_obj") or state.get("current_context")
        if ctx_obj and state.get("history") is not None:
            state["history"].record_transition(
                context=ctx_obj,
                start_time=state["context_start_time"],
                end_time=now,
            )
        state["context_start_time"] = now


def get_summary_times(state, now):
    """Return cumulative activity times including the ongoing active context duration."""
    times = dict(state["activity_time"])
    if state["last_context"] and state["context_start_time"]:
        active_dur = max(0.0, now - state["context_start_time"])
        times[state["last_context"]] = times.get(state["last_context"], 0) + active_dur
    return times


# ---------------- MONITOR THREAD ----------------
def monitor(state, lock):
    while True:
        try:
            current_window = get_active_window()
            now = time.time()

            # Check for active window change
            with lock:
                if current_window != state["last_window"]:
                    state["last_window"] = current_window
                    state["window_start_time"] = now
                    state["last_ocr_time"] = None
                    continue

                window_start = state["window_start_time"]
                last_ocr = state["last_ocr_time"]

            # Stable window check
            if window_start is not None and (now - window_start) >= STABILITY_TIME:
                is_first_ocr = last_ocr is None
                is_periodic_due = last_ocr is not None and (now - last_ocr) >= OCR_INTERVAL

                if is_first_ocr or is_periodic_due:
                    # Run capture and OCR in memory (outside lock to avoid blocking CLI)
                    try:
                        win_rect = get_active_window_rect()
                        screenshot = capture_screen(region=win_rect, save=False)
                        text = read_text_from_image(screenshot)
                    except Exception as e:
                        print(f"[MONITOR WARNING] Capture or OCR error: {e}")
                        text = ""
                        win_rect = None

                    activity, confidence = interpret_context(current_window, text)

                    # Build structured context representation
                    observation = RawObservation(
                        window_title=current_window,
                        ocr_text=text,
                        window_rect=win_rect,
                        timestamp=now,
                        metadata={"ocr_char_count": len(text)},
                    )
                    interpretation = InterpretedContext(
                        activity=activity,
                        confidence=confidence,
                    )
                    desktop_context = DesktopContext(
                        observation=observation,
                        interpretation=interpretation,
                        timestamp=now,
                    )

                    with lock:
                        state["last_ocr_time"] = now
                        state["current_context"] = desktop_context
                        context = activity

                        # FIRST time case
                        if state["last_context"] is None:
                            state["last_context"] = context
                            state["last_confidence"] = confidence
                            state["context_start_time"] = now
                            state["previous_context_obj"] = desktop_context

                            print("\n[CONTEXT UPDATE]")
                            print(f"You are {context}.")

                        # Candidate smoothing logic
                        elif context != state["last_context"]:
                            if state["candidate_context"] != context:
                                state["candidate_context"] = context
                                state["candidate_start_time"] = now
                            else:
                                if (
                                    state["candidate_start_time"] is not None
                                    and now - state["candidate_start_time"] >= MIN_CONTEXT_DURATION
                                ):
                                    # Close previous context episode and record transition in history
                                    if state["last_context"] and state["context_start_time"]:
                                        duration = now - state["context_start_time"]
                                        state["activity_time"][state["last_context"]] = (
                                            state["activity_time"].get(state["last_context"], 0) + duration
                                        )
                                        prev_ctx = state.get("previous_context_obj") or desktop_context
                                        if state.get("history") is not None:
                                            state["history"].record_transition(
                                                context=prev_ctx,
                                                start_time=state["context_start_time"],
                                                end_time=now,
                                            )

                                    state["context_start_time"] = now
                                    state["last_context"] = context
                                    state["last_confidence"] = confidence
                                    state["previous_context_obj"] = desktop_context

                                    print("\n[CONTEXT UPDATE]")
                                    print(f"You are {context}.")

                                    state["candidate_context"] = None
                                    state["candidate_start_time"] = None

                        elif context == state["last_context"]:
                            state["last_confidence"] = confidence
                            state["candidate_context"] = None
                            state["candidate_start_time"] = None
                            state["previous_context_obj"] = desktop_context

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"[MONITOR ERROR] Unexpected error in monitor loop: {e}")
            time.sleep(POLL_INTERVAL)


# ---------------- MAIN ----------------
def main():
    print("BLINDSPOT — quiet vision + OCR online\n")

    # shared state
    state = {
        "last_window": None,
        "last_context": None,
        "last_confidence": 0,
        "context_start_time": None,
        "window_start_time": None,
        "last_ocr_time": None,
        "activity_time": {},
        "candidate_context": None,
        "candidate_start_time": None,
        "current_context": None,
        "previous_context_obj": None,
        "history": ContextHistory(maxlen=50),
    }
    state_lock = threading.Lock()
    analyzer = ContextualUnderstandingAnalyzer()

    # start monitor in background
    threading.Thread(target=monitor, args=(state, state_lock), daemon=True).start()

    # command loop (MAIN THREAD — typing works)
    try:
        while True:
            cmd = input().strip().lower()
            now = time.time()

            if cmd == "status":
                print("\n[STATUS]")
                with state_lock:
                    if state["last_context"] and state["context_start_time"]:
                        current = now - state["context_start_time"]
                        print(f"current: {state['last_context']} ({format_time(current)})")
                    else:
                        print("no active context")

            elif cmd in ("context", "understanding"):
                with state_lock:
                    understanding = analyzer.analyze(
                        current_context=state["current_context"],
                        context_start_time=state["context_start_time"],
                        history=state["history"],
                        now=now,
                    )

                print("\n[CONTEXTUAL UNDERSTANDING]")
                if understanding.current_activity:
                    dur_str = format_time(understanding.current_duration)
                    print(f"Current:        {understanding.current_activity} ({dur_str}) | {understanding.current_window or ''}")
                    flow_str = " -> ".join(understanding.recent_activities) if understanding.recent_activities else (understanding.current_activity or "")
                    print(f"Recent Flow:    {flow_str}")
                    print(f"Pattern:        {understanding.pattern_type} (confidence: {understanding.confidence}/3)")
                    print(f"Interpretation: {understanding.interpretation}")
                else:
                    print("no active context detected")

            elif cmd == "history":
                with state_lock:
                    records = state["history"].get_recent(limit=10)
                    active_ctx = state["last_context"]
                    active_start = state["context_start_time"]
                    active_title = state["current_context"].window_title if state["current_context"] else ""

                print("\n[RECENT CONTEXT HISTORY]")
                if not records and not (active_ctx and active_start):
                    print("no history recorded yet")
                else:
                    for r in records:
                        t_start = format_timestamp(r.start_time)
                        t_end = format_timestamp(r.end_time) if r.end_time else "..."
                        print(f"{t_start} - {t_end} ({format_time(r.duration):>4}) | {r.activity:<20} | {r.window_title}")

                    if active_ctx and active_start:
                        current_dur = now - active_start
                        t_start = format_timestamp(active_start)
                        print(f"{t_start} - active   ({format_time(current_dur):>4}) | {active_ctx:<20} | {active_title}")

            elif cmd == "summary":
                with state_lock:
                    summary_times = get_summary_times(state, now)
                    summary_items = list(summary_times.items())

                print("\n[SUMMARY]")
                for activity, seconds in summary_items:
                    print(f"{activity}: {format_time(seconds)}")

    except (KeyboardInterrupt, EOFError):
        # close current context on exit
        now = time.time()
        with state_lock:
            close_active_context(state, now)
            summary_times = get_summary_times(state, now)
            final_items = list(summary_times.items())

        print("\n[FINAL TIME SUMMARY]")
        for activity, seconds in final_items:
            print(f"{activity}: {format_time(seconds)}")

        print("\nBLINDSPOT shutting down.")


if __name__ == "__main__":
    main()