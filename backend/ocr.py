from dataclasses import dataclass, field
import shutil
from typing import List, Optional, Tuple

import numpy as np
import pytesseract

# Import directly from your dedicated module
from preprocessing import preprocess_image

tesseract_path = shutil.which("tesseract") or "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

MEAN_CONF_THRESHOLD = 60.0
WORD_CONF_THRESHOLD = 50.0


@dataclass
class Token:
    word: str
    confidence: float
    is_low_conf: bool


@dataclass
class ScanResult:
    success: bool
    raw_text: str = ""
    reason: Optional[str] = None
    tokens: List[Token] = field(default_factory=list)
    has_low_conf: bool = False


def extract_words_with_confidence(
    processed_image: np.ndarray,
) -> List[Tuple[str, float]]:
    custom_config = r"--psm 6"
    data = pytesseract.image_to_data(
        processed_image, config=custom_config, output_type=pytesseract.Output.DICT
    )

    extracted_pairs = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = float(data["conf"][i])
        if word and conf != -1:
            extracted_pairs.append((word, conf))

    return extracted_pairs


def run_ocr(image_bytes: bytes) -> ScanResult:
    # Delegate vision pipeline to preprocessing module
    prep = preprocess_image(image_bytes)
    if not prep.success or prep.image is None:
        return ScanResult(success=False, reason=prep.reason)

    word_conf_pairs = extract_words_with_confidence(prep.image)
    if not word_conf_pairs:
        return ScanResult(success=False, reason="NO_TEXT_DETECTED")

    confidences = [score for _, score in word_conf_pairs]
    mean_confidence = float(np.mean(confidences))

    if mean_confidence < MEAN_CONF_THRESHOLD:
        return ScanResult(
            success=False,
            reason=f"LOW_CONFIDENCE (Mean: {mean_confidence:.1f}%)",
        )

    tokens = [
        Token(
            word=word,
            confidence=score,
            is_low_conf=(score < WORD_CONF_THRESHOLD),
        )
        for word, score in word_conf_pairs
    ]

    raw_text = " ".join([t.word for t in tokens])
    has_low_conf = any(t.is_low_conf for t in tokens)

    return ScanResult(
        success=True, raw_text=raw_text, tokens=tokens, has_low_conf=has_low_conf
    )