import time
import threading
from eyes.window import get_active_window
from eyes.screen import capture_screen
from eyes.ocr import read_text_from_image
from eyes.interpreter import interpret_context

STABILITY_TIME = 1.5  
MIN_CONTEXT_DURATION = 3  

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
    if state["last_context"] and state["context_start_time"]:
        duration = now - state["context_start_time"]
        state["activity_time"][state["last_context"]] = (
            state["activity_time"].get(state["last_context"], 0) + duration
        )
        state["context_start_time"] = now


# ---------------- MONITOR THREAD ----------------
def monitor(state):
    while True:
        try:
            current_window = get_active_window()
            now = time.time()

            # window change
            if current_window != state["last_window"]:
                state["last_window"] = current_window
                state["window_start_time"] = now
                state["ocr_done"] = False
                continue
            # allow periodic OCR refresh (important)
            if state["ocr_done"] and (now - state["window_start_time"] > 3):
                state["ocr_done"] = False

            # stable window → OCR once
            if (
                not state["ocr_done"]
                and state["window_start_time"] is not None
                and now - state["window_start_time"] >= STABILITY_TIME
            ):
                screenshot = capture_screen()
                text = read_text_from_image(screenshot)

                context, confidence = interpret_context(current_window, text)

                # FIRST time case
                if state["last_context"] is None:
                    state["last_context"] = context
                    state["last_confidence"] = confidence
                    state["context_start_time"] = now

                    print("\n[CONTEXT UPDATE]")
                    print(f"You are {context}.")

                # candidate smoothing logic
                elif context != state["last_context"]:
                    if state["candidate_context"] != context:
                        state["candidate_context"] = context
                        state["candidate_start_time"] = now
                    else:
                        if now - state["candidate_start_time"] >= MIN_CONTEXT_DURATION:
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
                state["ocr_done"] = True
                
            time.sleep(0.3)

        except KeyboardInterrupt:
            break


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
        "ocr_done": False,
        "activity_time": {},
        "candidate_context": None,
        "candidate_start_time": None,
    }

    # start monitor in background
    threading.Thread(target=monitor, args=(state,), daemon=True).start()

    # command loop (MAIN THREAD — typing works)
    try:
        while True:
            cmd = input().strip().lower()
            now = time.time()

            if cmd == "status":
                print("\n[STATUS]")
                if state["last_context"] and state["context_start_time"]:
                    current = now - state["context_start_time"]
                    print(f"current: {state['last_context']} ({format_time(current)})")
                else:
                    print("no active context")

            elif cmd == "summary":
                snapshot_current_context(state, now)

                print("\n[SUMMARY]")
                for activity, seconds in state["activity_time"].items():
                    print(f"{activity}: {format_time(seconds)}")


    except KeyboardInterrupt:
        # close current context on exit
        now = time.time()
        if state["last_context"] and state["context_start_time"]:
            duration = now - state["context_start_time"]
            state["activity_time"][state["last_context"]] = (
                state["activity_time"].get(state["last_context"], 0) + duration
            )

        print("\n[FINAL TIME SUMMARY]")
        for activity, seconds in state["activity_time"].items():
            print(f"{activity}: {format_time(seconds)}")

        print("\nBLINDSPOT shutting down.")


if __name__ == "__main__":
    main()
