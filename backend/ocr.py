from dataclasses import dataclass, field
import shutil
from typing import List, Optional, Tuple

import numpy as np
import pytesseract

# Import preprocessing pipeline
from preprocessing import preprocess_image, PreprocessResult

# Locate system executable path automatically
tesseract_path = shutil.which("tesseract") or "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Threshold Constants specified in NEA Success Criteria #5 & #6[span_1](start_span)[span_1](end_span)
MEAN_CONF_THRESHOLD = 90.0
WORD_CONF_THRESHOLD = 70.0

@dataclass
class Token:
   word: str
   confidence: float
   is_low_conf: bool

@dataclass
class OCRScanResult:
   success: bool
   raw_text: str = ""
   ingredients_text: str = ""
   trace_text: str = ""
   reason: Optional[str] = None
   tokens: List[Token] = field(default_factory=list)
   has_low_conf: bool = False

def extract_words_with_confidence(
   processed_image: np.ndarray,
) -> List[Tuple[str, float]]:
   """Extracts word tokens and per-word confidence scores using Tesseract image_to_data."""
   custom_config = r"--psm 6"
   data = pytesseract.image_to_data(
       processed_image, config=custom_config, output_type=pytesseract.Output.DICT
   )

   extracted_pairs = []
   num_boxes = len(data["text"])

   for i in range(num_boxes):
       word = data["text"][i].strip()
       conf = float(data["conf"][i])

       # Filter out empty tokens and non-word structural boxes (confidence == -1)
       if word and conf != -1:
           extracted_pairs.append((word, conf))
           
   return extracted_pairs

def parse_text(tokens: List[Token]) -> Tuple[str, str]:
   """Isolates the 'Ingredients:' block and 'May contain' trace advisory sections."""
   text = " ".join([token.word for token in tokens])
   # Simple section extraction logic based on keyword markers
   ingredients_text = ""
   trace_text = ""
   # Find ingredients section
   if "Ingredients:" in text:
       parts = text.split("Ingredients:")
       remainder = parts[1]
       # If trace marker exists within or after, split it out
       if "May contain" in remainder:
           ing_part, trace_part = remainder.split("May contain", 1)
           ingredients_text = ing_part.strip()
           trace_text = "May contain " + trace_part.strip()
       else:
           ingredients_text = remainder.strip()
   else:
       # Fallback if strict header is missing
       ingredients_text = text
   return ingredients_text, trace_text

def run_ocr(image_bytes: bytes) -> OCRScanResult:
   """Master controller implementing the OCR scanning pseudocode pipeline."""
   # 1. Run OpenCV Preprocessing Module
   prep: PreprocessResult = preprocess_image(image_bytes)
   if not prep.success or prep.image is None:
       return OCRScanResult(success=False, reason=prep.reason)
   # 2. Extract word-level data with confidence scores from Tesseract
   word_conf_pairs = extract_words_with_confidence(prep.image)
   if not word_conf_pairs:
       return OCRScanResult(success=False, reason="NO_TEXT_DETECTED")
   # 3. Calculate Mean Confidence against Success Criterion #5 (>= 90%)[span_2](start_span)[span_2](end_span)
   confidences = [score for _, score in word_conf_pairs]
   mean_confidence = float(np.mean(confidences))
   if mean_confidence < MEAN_CONF_THRESHOLD:
       return OCRScanResult(
           success=False,
           reason=f"LOW_CONFIDENCE (Mean: {mean_confidence:.1f}%)",
       )
   # 4. Build Token objects and flag low-confidence words (< 70%) per Criterion #6[span_3](start_span)[span_3](end_span)
   tokens = [
       Token(
           word=word,
           confidence=score,
           is_low_conf=(score < WORD_CONF_THRESHOLD),
       )
       for word, score in word_conf_pairs
   ]
   # 5. Parse text sections
   ingredients_text, trace_text = parse_text(tokens)
   if not ingredients_text:
       return OCRScanResult(success=False, reason="NO_INGREDIENTS_FOUND")
   raw_text = " ".join([t.word for t in tokens])
   has_low_conf = any(t.is_low_conf for t in tokens)
   return OCRScanResult(
       success=True,
       raw_text=raw_text,
       ingredients_text=ingredients_text,
       trace_text=trace_text,
       tokens=tokens,
       has_low_conf=has_low_conf,
   )

if __name__ == "__main__":
   # Test script for OCR module
   test_image_path = "test_image.jpg"
   try:
       with open(test_image_path, "rb") as image_file:
           file_bytes = image_file.read()
           result = run_ocr(file_bytes)
           print(f"OCR Scan Success: {result.success}")
           if result.success:
               print(f"Has Low Confidence Tokens: {result.has_low_conf}")
               print(f"Extracted Ingredients: {result.ingredients_text}")
               print(f"Extracted Trace Warning: {result.trace_text}")
           else:
               print(f"Failure Reason: {result.reason}")
   except FileNotFoundError:
       print(f"Test image not found at '{test_image_path}'. Add an image to test.")