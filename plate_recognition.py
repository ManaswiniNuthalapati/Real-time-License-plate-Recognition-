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
    Validate OCR output using structured plate-like formats.
    Rejects random words/logos.
    """
    if text is None:
        return False

    text = text.replace(" ", "").replace("\n", "").strip()

    # Common Indian license plate formats
    patterns = [
        r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',      # KA01AB1234
        r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}$' # KA1A1234, TS9EE456
    ]

    for p in patterns:
        if re.match(p, text):
            return True

    return False


def segment_chars(plate_img, fixed_width=400):
    """
    Character segmentation (not used directly now, kept for future upgrades)
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
        """
        Detect ALL possible plate-like regions (multiple cars support)
        """
        plates = []
        preprocessed = self.preprocess(input_img)
        possible_contours = self.extract_contours(preprocessed)

        for cnt in possible_contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Allow smaller plates (far cars)
            if w < 25 or h < 15:
                continue

            aspect_ratio = w / float(h)

            # Accept wide rectangular shapes
            if aspect_ratio < 1.0 or aspect_ratio > 10:
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

            aspect_ratio = w / float(h)
            if aspect_ratio < 1.0 or aspect_ratio > 10:
                continue

            gray_plate = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            gray_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)

            # Improve contrast
            gray_plate = cv2.equalizeHist(gray_plate)

            # OTSU threshold
            _, thresh = cv2.threshold(
                gray_plate, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            text = pytesseract.image_to_string(
                thresh,
                config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

            text = text.strip().replace(" ", "")

            # Must contain at least 2 digits
            if sum(c.isdigit() for c in text) < 2:
                continue

            if is_valid_plate_text(text):
                detected_texts.add(text)

    return detected_texts


# -------------------- VIDEO PROCESSING (REAL-TIME + MULTI-CAR) --------------------

def process_video_realtime(video_path, frame_placeholder, text_placeholder):
    findPlate = PlateFinder()
    cap = cv2.VideoCapture(video_path)

    detected_texts = set()
    ocr_cache = {}
    frame_count = 0
    SKIP_FRAMES = 2   # Fewer skips to catch fast-moving cars

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        frame_count += 1
        img = cv2.resize(img, (640, 360))

        # Skip fewer frames for better multi-car detection
        if frame_count % SKIP_FRAMES != 0:
            frame_placeholder.image(img, channels="BGR")
            continue

        possible_regions = findPlate.find_possible_plates(img)

        if possible_regions:
            for region, (x, y, w, h) in possible_regions:

                # Allow smaller plates
                if w < 25 or h < 15:
                    continue

                aspect_ratio = w / float(h)
                if aspect_ratio < 1.0 or aspect_ratio > 10:
                    continue

                roi_key = (x, y, w, h)
                text = None

                if roi_key in ocr_cache:
                    text = ocr_cache[roi_key]
                else:
                    gray_plate = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    gray_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)

                    # Improve contrast
                    gray_plate = cv2.equalizeHist(gray_plate)

                    # OTSU threshold
                    _, thresh = cv2.threshold(
                        gray_plate, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )

                    # Resize for faster OCR
                    thresh = cv2.resize(thresh, None, fx=0.7, fy=0.7)

                    text = pytesseract.image_to_string(
                        thresh,
                        config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    )

                    text = text.strip().replace(" ", "")

                    # Must contain digits
                    if sum(c.isdigit() for c in text) < 2:
                        text = None

                    if text and is_valid_plate_text(text):
                        ocr_cache[roi_key] = text
                    else:
                        text = None

                # Store all unique plates
                if text and text not in detected_texts:
                    detected_texts.add(text)

                if text:
                    cv2.rectangle(img, (x, y),
                                  (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.putText(img, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)

        frame_placeholder.image(img, channels="BGR")

        if detected_texts:
            text_placeholder.markdown("### 🔍 Detected Plates:")
            for t in detected_texts:
                text_placeholder.success(t)
        else:
            text_placeholder.info("Detecting license plates...")

    cap.release()
    return detected_texts
