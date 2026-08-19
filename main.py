import time
import threading
from eyes.window import get_active_window
from eyes.screen import capture_screen
from eyes.ocr import read_text_from_image
from eyes.interpreter import interpret_context

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


def snapshot_current_context(state, now):
    """Accumulate ongoing context duration into activity_time and advance context_start_time."""
    if state["last_context"] and state["context_start_time"]:
        duration = now - state["context_start_time"]
        state["activity_time"][state["last_context"]] = (
            state["activity_time"].get(state["last_context"], 0) + duration
        )
        state["context_start_time"] = now


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
                        screenshot = capture_screen(save=False)
                        text = read_text_from_image(screenshot)
                    except Exception as e:
                        print(f"[MONITOR WARNING] Capture or OCR error: {e}")
                        text = ""

                    context, confidence = interpret_context(current_window, text)

                    with lock:
                        state["last_ocr_time"] = now

                        # FIRST time case
                        if state["last_context"] is None:
                            state["last_context"] = context
                            state["last_confidence"] = confidence
                            state["context_start_time"] = now

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
                                    # close previous context
                                    if state["last_context"] and state["context_start_time"]:
                                        duration = now - state["context_start_time"]
                                        state["activity_time"][state["last_context"]] = (
                                            state["activity_time"].get(state["last_context"], 0) + duration
                                        )

                                    state["context_start_time"] = now
                                    state["last_context"] = context
                                    state["last_confidence"] = confidence

                                    print("\n[CONTEXT UPDATE]")
                                    print(f"You are {context}.")

                                    state["candidate_context"] = None
                                    state["candidate_start_time"] = None

                        elif context == state["last_context"]:
                            state["last_confidence"] = confidence
                            state["candidate_context"] = None
                            state["candidate_start_time"] = None

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
    }
    state_lock = threading.Lock()

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

            elif cmd == "summary":
                with state_lock:
                    snapshot_current_context(state, now)
                    summary_items = list(state["activity_time"].items())

                print("\n[SUMMARY]")
                for activity, seconds in summary_items:
                    print(f"{activity}: {format_time(seconds)}")

    except (KeyboardInterrupt, EOFError):
        # close current context on exit
        now = time.time()
        with state_lock:
            snapshot_current_context(state, now)
            final_items = list(state["activity_time"].items())

        print("\n[FINAL TIME SUMMARY]")
        for activity, seconds in final_items:
            print(f"{activity}: {format_time(seconds)}")

        print("\nBLINDSPOT shutting down.")


if __name__ == "__main__":
    main()