import time
from mss import mss
from PIL import Image
from eyes.window import get_active_window_rect


def capture_screen(region=None, save=False):
    """
    Capture the active window's bounding rectangle or a specified custom region.
    Falls back to the central 60% of the primary monitor if active window rect is unavailable or invalid.
    """
    with mss() as sct:
        target_region = None

        if region is not None:
            target_region = region
        else:
            target_region = get_active_window_rect()

        # Validate region or fallback to central 60% of primary monitor
        if not target_region or target_region.get("width", 0) <= 0 or target_region.get("height", 0) <= 0:
            monitor = sct.monitors[1]
            width = monitor["width"]
            height = monitor["height"]
            left = int(width * 0.2)
            top = int(height * 0.2)
            right = int(width * 0.8)
            bottom = int(height * 0.8)

            target_region = {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
            }

        try:
            screenshot = sct.grab(target_region)
        except Exception:
            # Fallback to central region if grabbing window rect raises (e.g. coordinates out of bounds)
            monitor = sct.monitors[1]
            fallback_region = {
                "left": int(monitor["width"] * 0.2),
                "top": int(monitor["height"] * 0.2),
                "width": int(monitor["width"] * 0.6),
                "height": int(monitor["height"] * 0.6),
            }
            screenshot = sct.grab(fallback_region)

        img = Image.frombytes(
            "RGB",
            screenshot.size,
            screenshot.rgb,
        )

        if save:
            filename = f"screenshot_{int(time.time())}.png"
            img.save(filename)
            return filename

        return img