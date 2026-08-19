import pytesseract
import cv2
import os
import re
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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


def read_text_from_image(image_path, cleanup=True):
    img = cv2.imread(image_path)
    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    raw_text = pytesseract.image_to_string(gray, lang="eng")
    text = clean_text(raw_text)

    if cleanup and os.path.exists(image_path):
        os.remove(image_path)

    return text
