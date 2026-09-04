import cv2
import pytesseract
import shutil
import numpy as np

#locate system executable path automatically
tesseract_path = shutil.which("tesseract") or "/usr/bin/tesseract"

#assign binary path to pytesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_path