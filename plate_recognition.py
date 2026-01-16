import cv2
import numpy as np
import pytesseract
import re

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------- TEXT VALIDATION --------------------

def is_valid_plate_text(text):
    if not text:
        return False

    text = text.upper().replace(" ", "").replace("\n", "").strip()

    # Indian plate formats (flexible)
    patterns = [
        r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}$',  # MH20EE7598
        r'^[A-Z]{2}\d{1,2}\d{3,4}$',           # TS098765
        r'^[A-Z0-9]{6,12}$'
    ]

    for p in patterns:
        if re.match(p, text):
            return True
    return False


# -------------------- PLATE DETECTION --------------------

def detect_plates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_regions = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.018 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Rectangle shape
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 100 and h > 30:
                plate_regions.append((x, y, w, h))

    return plate_regions


# -------------------- OCR FUNCTION --------------------

def read_text_from_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(
        gray,
        config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    text = text.replace(" ", "").replace("\n", "").strip()
    return text


# -------------------- IMAGE PROCESSING --------------------

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("❌ Could not load image.")

    detected_texts = set()
    plates = detect_plates(img)

    for (x, y, w, h) in plates:
        plate_img = img[y:y+h, x:x+w]
        text = read_text_from_plate(plate_img)

        if is_valid_plate_text(text):
            detected_texts.add(text)

    return detected_texts


# -------------------- VIDEO PROCESSING --------------------

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detected_texts = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        plates = detect_plates(frame)

        for (x, y, w, h) in plates:
            plate_img = frame[y:y+h, x:x+w]
            text = read_text_from_plate(plate_img)

            if is_valid_plate_text(text):
                detected_texts.add(text)

    cap.release()
    return detected_texts
