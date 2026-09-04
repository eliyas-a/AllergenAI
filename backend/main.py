from fastapi import FastAPI
import pytesseract
import shutil

tesseract_path = shutil.which("tesseract") or "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

app = FastAPI()

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "tesseract_version": str(pytesseract.get_tesseract_version()),
        "binary_path": tesseract_path
    }