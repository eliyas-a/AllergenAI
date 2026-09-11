import cv2
import pytesseract
import shutil
import numpy as np

#locate system executable path automatically
tesseract_path = shutil.which("tesseract") or "/usr/bin/tesseract"

#assign binary path to pytesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_path

def convert_to_greyscale(image: np.ndarray): ## converts BGR image array to a 1-channel greyscale
    ## use cv2.cvtColor
    pass

def remove_noise(gray_image: np.ndarray): ## applies slight Gausssian blur to smooth camera grain
    ## use cv2.GaussianBlur
    pass

def apply_threshold(blurred_image: np.ndarray): ## applies binarization to maximise text contrast
    ## use cv2.threshold with cv2.THRESH_BINARY + cv2.THRESH_OTSU
    pass

def extract_text_from_bytes(binary_image: bytes):
    # convert raw bytes to a NumPy array
    np_array = np.frombuffer(image_bytes, np.uint8) 

    #decode array into openCV image matrix
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image ")

    # pass through preprocessing pipeline
    gray = convert_to_greyscale(image)
    blurred = remove_noise(gray)
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