import cv2
import numpy as np
from skimage import measure
import imutils
import pytesseract
import re

# If on Windows, set this path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------- Helper Functions --------------------

def is_valid_plate_text(text):
    """
    Validate extracted OCR text.
    Accept only alphanumeric strings of reasonable length.
    """
    if text is None:
        return False

    text = text.replace(" ", "").replace("\n", "").strip()

    # Accept only alphanumeric strings between 4 and 10 characters
    if re.match(r'^[A-Z0-9]{4,10}$', text):
        return True
    return False


def segment_chars(plate_img, fixed_width=400):
    """
    Segment possible characters from plate image
    """
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]
    thresh = cv2.adaptiveThreshold(V, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   11, 2)
    thresh = cv2.bitwise_not(thresh)

    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    labels = measure.label(thresh, background=0)
    charCandidates = np.zeros(thresh.shape, dtype='uint8')

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255
        cnts = cv2.findContours(labelMask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])

            # Loose character constraints (not strict)
            if aspectRatio < 1.0 and solidity > 0.1 and heightRatio > 0.4:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    contours, _ = cv2.findContours(charCandidates,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        characters = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            temp = bgr_thresh[y:y + h, x:x + w]
            characters.append(temp)
        return characters
    else:
        return None


# -------------------- Plate Detection Class --------------------

class PlateFinder:
    def __init__(self):
        self.element_structure = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        sobelX = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, threshold_img = cv2.threshold(sobelX, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph_img = cv2.morphologyEx(threshold_img,
                                     cv2.MORPH_CLOSE,
                                     self.element_structure)
        return morph_img

    def extract_contours(self, after_preprocess):
        contours, _ = cv2.findContours(after_preprocess,
                                       mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_NONE)
        return contours

    def find_possible_plates(self, input_img):
        plates = []
        preprocessed = self.preprocess(input_img)
        possible_contours = self.extract_contours(preprocessed)

        for cnt in possible_contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Only skip extremely small regions (noise)
            if w < 40 or h < 20:
                continue

            region = input_img[y:y + h, x:x + w]
            plates.append((region, (x, y, w, h)))

        return plates if plates else None


# -------------------- IMAGE PROCESSING --------------------

def process_image(image_path):
    """
    Process a single image and return detected license plate text
    """
    findPlate = PlateFinder()
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("❌ Image not loaded. Please upload a valid image file.")

    detected_texts = set()
    possible_regions = findPlate.find_possible_plates(img)

    if possible_regions:
        for region, (x, y, w, h) in possible_regions:
            gray_plate = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            gray_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)

            text = pytesseract.image_to_string(
                gray_plate,
                config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

            text = text.strip().replace(" ", "")

            # FINAL VALIDATION: only accept valid plate-like text
            if is_valid_plate_text(text):
                detected_texts.add(text)

    return detected_texts


# -------------------- VIDEO PROCESSING --------------------

def process_video(video_path):
    """
    Process a video file and return:
    1. List of frames with detected plates drawn
    2. Set of detected license plate texts
    """
    findPlate = PlateFinder()
    cap = cv2.VideoCapture(video_path)

    frames = []
    detected_texts = set()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        possible_regions = findPlate.find_possible_plates(img)

        if possible_regions:
            for region, (x, y, w, h) in possible_regions:
                gray_plate = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                gray_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)

                text = pytesseract.image_to_string(
                    gray_plate,
                    config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )

                text = text.strip().replace(" ", "")

                if is_valid_plate_text(text):
                    detected_texts.add(text)

                    # Draw rectangle only for valid plates
                    cv2.rectangle(img, (x, y),
                                  (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.putText(img, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)

        frames.append(img)

    cap.release()
    return frames, detected_texts
