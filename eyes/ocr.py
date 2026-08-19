import os
import re
import shutil
import numpy as np
from PIL import Image
import cv2
import pytesseract


def find_tesseract_binary():
    """Locate the Tesseract executable via env var, PATH, or standard installation paths."""
    # 1. Explicit environment variable
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and os.path.isfile(env_cmd):
        return env_cmd

    # 2. Check system PATH
    which_cmd = shutil.which("tesseract") or shutil.which("tesseract.exe")
    if which_cmd:
        return which_cmd

    # 3. Standard Windows installation directories
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return path

    return None


# Initialize Tesseract path on module load if found
_tesseract_path = find_tesseract_binary()
if _tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_path


def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        # remove empty lines
        if not line:
            continue

        # remove very short junk
        if len(line) <= 2:
            continue

        # remove lines full of symbols
        if re.fullmatch(r"[\W_]+", line):
            continue

        # collapse spaces
        line = re.sub(r"\s+", " ", line)

        cleaned.append(line)

    return "\n".join(cleaned)


def read_text_from_image(image_input, cleanup=True):
    """
    Extract text from an image. Supports PIL Image, numpy ndarray, or file path string.
    """
    global _tesseract_path
    if not _tesseract_path:
        _tesseract_path = find_tesseract_binary()
        if _tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = _tesseract_path
        else:
            print("[OCR WARNING] Tesseract-OCR not found. Please install Tesseract or set TESSERACT_CMD.")
            return ""

    file_to_cleanup = None

    try:
        if isinstance(image_input, Image.Image):
            # In-memory PIL Image -> OpenCV grayscale
            np_img = np.array(image_input)
            if len(np_img.shape) == 2:
                gray = np_img
            elif np_img.shape[2] == 4:
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                gray = image_input
            elif image_input.shape[2] == 4:
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)

        elif isinstance(image_input, (str, os.PathLike)):
            image_path = str(image_input)
            if cleanup:
                file_to_cleanup = image_path

            img = cv2.imread(image_path)
            if img is None:
                return ""
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            return ""

        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

        raw_text = pytesseract.image_to_string(gray, lang="eng")
        return clean_text(raw_text)

    except pytesseract.TesseractNotFoundError:
        print("[OCR WARNING] Tesseract executable not found. Please verify Tesseract installation.")
        return ""
    except Exception as e:
        print(f"[OCR ERROR] Failed to process image for OCR: {e}")
        return ""
    finally:
        if file_to_cleanup and os.path.exists(file_to_cleanup):
            try:
                os.remove(file_to_cleanup)
            except OSError:
                pass