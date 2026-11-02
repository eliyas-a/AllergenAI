from dataclasses import dataclass, field
import re
import shutil
from typing import List, Optional, Tuple

import numpy as np
import pytesseract

# Import preprocessing pipeline
from preprocessing import preprocess_image, PreprocessResult

# Locate system executable path automatically
tesseract_path = shutil.which("tesseract") or "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

### CONSTANTS ###
# Currently lowered to 60 and 40 for testing – PENDING POSSIBLE CHANGE
MEAN_CONF_THRESHOLD = 60.0
WORD_CONF_THRESHOLD = 40.0

INGREDIENTS_PATTERN = re.compile(r"\bingredients\b\s*:?", re.IGNORECASE)
TRACE_PATTERN = re.compile(r"\b(may\s+contain|traces?\s+of)\b", re.IGNORECASE)


@dataclass
class Token:
    word: str
    confidence: float
    is_low_conf: bool  # True if confidence < WORD_CONF_THRESHOLD (pending change)


@dataclass
class ScanResult:
    """Base class for any scan stage that can succeed or fail (matches design 2.2.5)."""
    success: bool
    reason: Optional[str] = None


@dataclass
class OCRScanResult(ScanResult):
    raw_text: str = ""
    ingredients_text: str = ""
    trace_text: str = ""
    tokens: List[Token] = field(default_factory=list)
    has_low_conf: bool = False
    mean_confidence: float = 0.0
    low_mean_flag: bool = False  # True if mean is below MEAN_CONF_THRESHOLD


def extract_words_with_confidence(
    processed_image: np.ndarray,
) -> List[Tuple[str, float]]:
    """Extracts word tokens and per-word confidence scores using Tesseract image_to_data."""
    custom_config = r"--psm 6"
    data = pytesseract.image_to_data(
        processed_image, config=custom_config, output_type=pytesseract.Output.DICT
    )

    extracted_pairs = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = float(data["conf"][i])

        # Skip empty tokens and structural boxes (confidence == -1)
        if word and conf != -1:
            extracted_pairs.append((word, conf))

    return extracted_pairs


def parse_text(text: str) -> Tuple[str, str]:
    """
    Splits OCR text into (ingredients_text, trace_text).
    If no 'Ingredients' header is found, returns an empty ingredients string.
    """
    
    header = INGREDIENTS_PATTERN.search(text)
    if header is None:
        return "", ""

    trace = TRACE_PATTERN.search(text)

    if trace is None:
        return text[header.end():].strip(), ""

    if trace.start() >= header.end():
        # Usual layout: ingredients list, then trace warning
        ingredients_text = text[header.end():trace.start()]
        trace_text = text[trace.start():]
    else:
        # Trace warning printed before the ingredients header
        ingredients_text = text[header.end():]
        trace_text = text[trace.start():header.start()]

    return ingredients_text.strip(), trace_text.strip()


def run_ocr(image_bytes: bytes) -> OCRScanResult:
    """Master controller implementing the OCR scanning pseudocode pipeline."""

    # 1. Run OpenCV preprocessing module
    prep: PreprocessResult = preprocess_image(image_bytes)
    if not prep.success or prep.image is None:
        return OCRScanResult(success=False, reason=prep.reason)

    # 2. Extract word-level data with confidence scores from Tesseract
    word_conf_pairs = extract_words_with_confidence(prep.image)
    if not word_conf_pairs:
        return OCRScanResult(success=False, reason="NO_TEXT_DETECTED")

    # 3. Mean confidence (Success Criterion #5)
    mean_confidence = float(np.mean([score for _, score in word_conf_pairs]))
    low_mean_flag = mean_confidence < MEAN_CONF_THRESHOLD

    # TODO: RESTORE FAILURE ON LOW MEAN CONFIDENCE BEFORE FINAL VERSION.
    # reason="LOW_CONFIDENCE" AND PROMPT A RETAKE
    # IF LEFT AS IS, DOCUMENT AS A DESIGN CHANGE.

    # 4. Build Token objects and flag low-confidence words (Criterion #6)
    tokens = [
        Token(word=word, confidence=score, is_low_conf=(score < WORD_CONF_THRESHOLD))
        for word, score in word_conf_pairs
    ]

    # 5. Join once and parse; no header means failure, never a false 'Safe'
    raw_text = " ".join(t.word for t in tokens)
    ingredients_text, trace_text = parse_text(raw_text)
    if not ingredients_text:
        return OCRScanResult(success=False, reason="NO_INGREDIENTS_FOUND")

    return OCRScanResult(
        success=True,
        raw_text=raw_text,
        ingredients_text=ingredients_text,
        trace_text=trace_text,
        tokens=tokens,
        has_low_conf=any(t.is_low_conf for t in tokens),
        mean_confidence=mean_confidence,
        low_mean_flag=low_mean_flag,
    )


if __name__ == "__main__":
    # Test script for OCR module
    test_image_path = "test_image.jpg"

    try:
        with open(test_image_path, "rb") as image_file:
            result = run_ocr(image_file.read())

            print(f"OCR Scan Success: {result.success}")
            if result.success:
                print(f"Mean Word Confidence: {result.mean_confidence:.1f}%")
                print(f"Below Mean Threshold ({MEAN_CONF_THRESHOLD}): {result.low_mean_flag}")
                print(f"Has Low Confidence Tokens: {result.has_low_conf}")
                print(f"Extracted Ingredients: {result.ingredients_text}")
                print(f"Extracted Trace Warning: {result.trace_text}")
            else:
                print(f"Failure Reason: {result.reason}")

    except FileNotFoundError:
        print(f"Test image not found at '{test_image_path}'. Add an image to test.")