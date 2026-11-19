from fastapi import FastAPI
import pytesseract
import shutil

# Locate the Tesseract binary on PATH
# Needed because pytesseract only wraps the binary, it doesn't ship it
tesseract_path = shutil.which("tesseract") or "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

app = FastAPI()

@app.get("/health")
def health_check():
    return {
        "status": "online",
        
        # str() is essential, would normally return a Version object
        # FastAPI cannot serialise to JSON
        "tesseract_version": str(pytesseract.get_tesseract_version()),
        "binary_path" : tesseract_path
    }