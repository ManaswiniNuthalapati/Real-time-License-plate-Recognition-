import cv2
import numpy as np
from skimage import measure
import imutils
import pytesseract

# If on Windows, set this path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------- Helper Functions --------------------

def sort_cont(character_contours):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours,
                                                          boundingBoxes),
                                                      key=lambda b: b[1][i],
                                                      reverse=False))
    return character_contours


def segment_chars(plate_img, fixed_width):
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

            if aspectRatio < 1.0 and solidity > 0.15 and \
               heightRatio > 0.5 and heightRatio < 0.95 and boxW > 14:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    contours, hier = cv2.findContours(charCandidates,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sort_cont(contours)
        addPixel = 4
        characters = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            y = max(0, y - addPixel)
            x = max(0, x - addPixel)
            temp = bgr_thresh[y:y + h + addPixel*2,
                              x:x + w + addPixel*2]
            characters.append(temp)
        return characters
    else:
        return None


# -------------------- Plate Detection Class --------------------

class PlateFinder:
    def __init__(self, minPlateArea, maxPlateArea):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea
        self.element_structure = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        sobelX = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        ret2, threshold_img = cv2.threshold(sobelX, 0, 255,
                                            cv2.THRESH_BINARY +
                                            cv2.THRESH_OTSU)
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img,
                         op=cv2.MORPH_CLOSE,
                         kernel=self.element_structure,
                         dst=morph_n_thresholded_img)
        return morph_n_thresholded_img

    def extract_contours(self, after_preprocess):
        contours, _ = cv2.findContours(after_preprocess,
                                       mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray,
                                       255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       11, 2)
        contours, _ = cv2.findContours(thresh.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            if not self.ratioCheck(max_cntArea, plate.shape[1], plate.shape[0]):
                return plate, False, None
            return plate, True, [x, y, w, h]
        else:
            return plate, False, None

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(
                after_validation_img)
            if plateFound:
                characters_on_plate = self.find_characters_on_plate(
                    after_clean_plate_img)
                if (characters_on_plate is not None and
                    len(characters_on_plate) >= 1):
                    x1, y1, w1, h1 = coordinates
                    coordinates = x1 + x, y1 + y
                    return after_clean_plate_img, characters_on_plate, coordinates
        return None, None, None

    def find_possible_plates(self, input_img):
        plates = []
        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(self.after_preprocess)
        for cnts in possible_plate_contours:
            plate, characters_on_plate, coordinates = self.check_plate(input_img, cnts)
            if plate is not None:
                plates.append((plate, characters_on_plate, coordinates))
        return plates if plates else None

    def find_characters_on_plate(self, plate):
        return segment_chars(plate, 400)

    def ratioCheck(self, area, width, height):
        ratio = float(width) / float(height if height else 1)
        return (area >= self.min_area and area <= self.max_area and
                (ratio >= 3 and ratio <= 6))

    def preRatioCheck(self, area, width, height):
        ratio = float(width) / float(height if height else 1)
        return (area >= self.min_area and area <= self.max_area and
                (ratio >= 2.5 and ratio <= 7))

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect
        if width < height:
            angle = 90 + rect_angle
        else:
            angle = -rect_angle
        if angle > 15 or height == 0 or width == 0:
            return False
        area = width * height
        return self.preRatioCheck(area, width, height)


# -------------------- IMAGE PROCESSING --------------------

def process_image(image_path):
    """
    Process a single image and return detected license plate text
    """
    findPlate = PlateFinder(minPlateArea=1000, maxPlateArea=50000)
    img = cv2.imread(image_path)

    # 🚨 Safety check
    if img is None:
        raise ValueError("❌ Image not loaded. Please upload a valid image file.")

    detected_texts = set()

    possible_plates = findPlate.find_possible_plates(img)

    if possible_plates:
        for plate_img, characters, coord in possible_plates:
            gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            gray_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)

            text = pytesseract.image_to_string(
                gray_plate,
                config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

            if text.strip():
                detected_texts.add(text.strip())

    return detected_texts


# -------------------- VIDEO PROCESSING --------------------

def process_video(video_path):
    """
    Process a video file and return:
    1. List of frames with detected plates drawn
    2. Set of detected license plate texts
    """
    findPlate = PlateFinder(minPlateArea=1000, maxPlateArea=50000)
    cap = cv2.VideoCapture(video_path)

    frames = []
    detected_texts = set()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        possible_plates = findPlate.find_possible_plates(img)

        if possible_plates:
            for plate_img, characters, coord in possible_plates:
                x, y = coord
                cv2.rectangle(img, (x, y),
                              (x + plate_img.shape[1], y + plate_img.shape[0]),
                              (0, 255, 0), 2)

                gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                gray_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)

                text = pytesseract.image_to_string(
                    gray_plate,
                    config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )

                if text.strip():
                    detected_texts.add(text.strip())

        frames.append(img)

    cap.release()
    return frames, detected_texts
