from mss import mss
from PIL import Image
import time

def capture_screen(save=True):
    with mss() as sct:
        monitor = sct.monitors[1]

        width = monitor["width"]
        height = monitor["height"]
        left = int(width * 0.2)
        top = int(height * 0.2)
        right = int(width * 0.8)
        bottom = int(height * 0.8)

        region = {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top
        }

        screenshot = sct.grab(region)

        img = Image.frombytes(
            "RGB",
            screenshot.size,
            screenshot.rgb
        )

        if save:
            filename = f"screenshot_{int(time.time())}.png"
            img.save(filename)
            return filename

        return img
