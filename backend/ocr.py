from dataclasses import dataclass, field
import shutil
from typing import, List, Optional, Tuple

import cv2
import pytesseract
import numpy as np

#configure Tesseract binary path
tesseract_path = shutil.which("tesseract") or "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

#Threshold Constants
MEAN_CONF_THRESHOLD = 60.0
WORD_CONF_THRESHOLD = 50.0

# ---Data Structures---
@dataclass
class Token:
    word: str
    confidence: float
    is_low_conf: bool

@dataclass
class PreprocessResult:
    success: bool
    image: Optional[np.ndarray] = None
    reason: Optional[str] = None

@dataclass

# ---Preprocessing Pipeline--- #
def convert_to_greyscale(image: np.ndarray): ## converts BGR image array to a 1-channel greyscale
    """"Converts a BGR image array into greyscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(grey_image: np.ndarray): 
    """Applies Gaussian blur to smooth camera grain"""
    return cv2.GaussianBlur(grey_image, (3,3), 0)


def apply_threshold(blurred_image: np.ndarray):
    """Applies Otsu's binarisation to maximise text contrast"""
    # returns only thresholded image matrix
    _, thresh = cv2.threshold(
        blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


######GOT UP TO HERE####
#######
#######
#####
    
def extract_text_from_bytes(binary_image: bytes):
    # convert raw bytes to a NumPy array
    np_array = np.frombuffer(image_bytes, np.uint8) 

    #decode array into openCV image matrix
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image ")

    # pass through preprocessing pipeline
    grey = convert_to_greyscale(image)
    blurred = remove_noise(grey)
    thresh = apply_threshold(blurred)

    # configure Tesseract parameter
    custom_config = r"--psm 6"

    # execute OCR
    extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

    return extracted_text.strip()

if __name__ == "__main__":
    test_image_path = "... .jpg"

    try:
        with open(test_image_path, "rb") as image_file:
            file_bytes = image_file.read()
            text = extract_text_from_bytes(file_bytes)
            print("--- Extracted Text ---")
            print(text)

    except FileNotFoundError:
        print(f"Test image not found at '{test_image_path}'. Add an image to test.")