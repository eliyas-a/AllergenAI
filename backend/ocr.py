from dataclasses import dataclass, field
from typing import List, Optional
import re

import numpy as np
import pytesseract

from preprocessing import preprocess_image, TEST_IMAGE


### CONSTANTS ###

MEAN_CONF_THRESHOLD = 50.0
WORD_CONF_THRESHOLD = 40.0
TESSERACT_CONFIG = "--oem 3 --psm 6"


# Matches an ingredients heading at the beginning of a line.
# The colon is optional, but the line anchor prevents phrases such as
# "see ingredients in bold" from being mistaken for a heading.
INGREDIENTS_PATTERN = re.compile(
    r"^[ \t]*ingredients\b\s*:?\s*",
    re.IGNORECASE | re.MULTILINE,
)


# Matches common trace/allergen advisory formulations.
TRACE_PATTERN = re.compile(
    r"\b("
    r"may\s+(?:also\s+)?contain"
    r"|traces?\s+of"
    r"|produced\s+in\s+a\s+(?:factory|facility)\s+"
    r"that\s+(?:also\s+)?handles"
    r"|made\s+in\s+a\s+(?:factory|facility)\s+"
    r"that\s+(?:also\s+)?handles"
    r"|manufactured\s+in\s+a\s+(?:factory|facility)\s+"
    r"that\s+(?:also\s+)?handles"
    r"|made\s+on\s+(?:shared\s+)?equipment\s+"
    r"that\s+(?:also\s+)?processes"
    r"|manufactured\s+on\s+(?:shared\s+)?equipment"
    r"|produced\s+on\s+(?:shared\s+)?equipment"
    r")\b[^.]*\.?",
    re.IGNORECASE,
)


# Section headings that indicate the ingredients list has finished.
# Anchoring these to the beginning of a line prevents ordinary ingredient
# text from being mistaken for a new section.
END_PATTERN = re.compile(
    r"^[ \t]*("
    r"nutrition\w*"
    r"|typical\s+values"
    r"|storage"
    r"|store\s+in"
    r"|best\s+before"
    r"|use\s+by"
    r"|directions"
    r"|manufactured\s+by"
    r"|ingr\w*\s*:"
    r"|contains\s*:"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


### DATA CLASSES ###

@dataclass
class Token:
    word: str
    confidence: float
    is_low_conf: bool
    line_id: tuple


@dataclass
class OCRScanResult:
    success: bool
    reason: Optional[str] = None
    raw_text: str = ""
    ingredients: List[str] = field(default_factory=list)
    trace_statements: List[str] = field(default_factory=list)
    tokens: List[Token] = field(default_factory=list)
    mean_confidence: float = 0.0
    has_low_conf: bool = False


### TOKEN EXTRACTION ###

def extract_tokens(image: np.ndarray) -> List[Token]:
    """Runs Tesseract and returns recognised words with confidence scores."""

    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    tokens = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        confidence = float(data["conf"][i])

        # Ignore empty entries and layout boxes without recognised words.
        if word and confidence >= 0:
            line_id = (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i],
            )

            tokens.append(
                Token(
                    word=word,
                    confidence=confidence,
                    is_low_conf=confidence < WORD_CONF_THRESHOLD,
                    line_id=line_id,
                )
            )

    return tokens


### TEXT RECONSTRUCTION ###

def build_text_with_spans(tokens: List[Token]):
    """
    Reconstructs the OCR text and records the character positions
    occupied by each token.

    The spans allow confidence scores to be associated with the
    ingredients and trace sections rather than unrelated text.
    """

    parts = []
    token_spans = []
    position = 0

    for i, token in enumerate(tokens):

        if i > 0:
            same_line = token.line_id == tokens[i - 1].line_id
            separator = " " if same_line else "\n"

            parts.append(separator)
            position += len(separator)

        start = position

        parts.append(token.word)
        position += len(token.word)

        token_spans.append((start, position, token))

    return "".join(parts), token_spans


### SECTION PARSING ###

def parse_text_with_ranges(text: str):
    """
    Extracts the ingredients section and trace statements.

    Also returns character ranges identifying the relevant text,
    allowing confidence calculations to exclude unrelated packaging text.
    """

    ingredients_text = ""
    trace_statements = []
    relevant_ranges = []

    header = INGREDIENTS_PATTERN.search(text)

    if header:

        start = header.end()
        end = len(text)

        # Find the next section heading.
        end_match = END_PATTERN.search(text, start)

        if end_match:
            end = min(end, end_match.start())

        # The ingredients section ends before the first trace statement.
        trace_match = TRACE_PATTERN.search(text, start)

        if trace_match:
            end = min(end, trace_match.start())

        ingredients_text = text[start:end].strip()

        # Skip leading whitespace so the range matches the stripped text.
        if ingredients_text:
            actual_start = start + len(text[start:end]) - len(
                text[start:end].lstrip()
            )

            relevant_ranges.append(
                (actual_start, actual_start + len(ingredients_text))
            )

    # Extract trace statements separately from the ingredients list.
    for match in TRACE_PATTERN.finditer(text):

        statement = cut_at_section_end_of_trace(match.group(0))

        if statement:
            trace_statements.append(statement)

            # Locate the statement within the original reconstructed text.
            statement_start = match.start()
            statement_end = statement_start + len(statement)

            relevant_ranges.append((statement_start, statement_end))

    return ingredients_text, trace_statements, relevant_ranges


def cut_at_section_end_of_trace(statement: str) -> str:
    """Removes section headings accidentally included in a trace statement."""

    match = END_PATTERN.search(statement)

    if match:
        statement = statement[:match.start()]

    return statement.strip()


### INGREDIENT SPLITTING ###

def split_ingredients(ingredients_text: str) -> List[str]:
    """
    Splits ingredients into individual terms.

    Main ingredients and parenthetical sub-ingredients are retained
    separately. For example:

    Vegetable oil (palm oil, sunflower oil)

    becomes:
    ["vegetable oil", "palm oil", "sunflower oil"]
    """

    text = ingredients_text.replace("\n", " ")

    # Remove percentages, e.g. "Sugar 12.5%".
    text = re.sub(r"\d+(?:\.\d+)?\s*%", "", text)

    # Some labels separate ingredients with bullets or semicolons, not commas.
    text = re.sub(r"[•·●▪;]", ",", text)

    def parse_level(value: str) -> List[str]:

        main_parts = []
        parenthetical_groups = []

        current = []
        group_buffer = []
        bracket_stack = []

        opening_brackets = "( [".replace(" ", "")
        closing_brackets = ")]"

        for char in value:

            # An opening bracket starts a new parenthetical group.
            if char in opening_brackets:

                if not bracket_stack:
                    group_buffer = []
                else:
                    group_buffer.append(char)

                bracket_stack.append(char)

            elif char in closing_brackets and bracket_stack:
                bracket_stack.pop()

                if not bracket_stack:
                    parenthetical_groups.append(
                        "".join(group_buffer)
                    )
                else:
                    group_buffer.append(char)

            elif bracket_stack:
                group_buffer.append(char)

            elif char in closing_brackets:
                # Ignore unmatched closing brackets.
                continue

            elif char == ",":
                main_parts.append("".join(current))
                current = []

            else:
                current.append(char)

        if current:
            main_parts.append("".join(current))

        # Preserve the contents of an unclosed parenthetical group.
        if bracket_stack and group_buffer:
            parenthetical_groups.append("".join(group_buffer))

        ingredients = []

        # Add the main ingredients.
        for part in main_parts:

            cleaned = part.strip(" .*-–—:;()[]").lower()

            if cleaned:
                ingredients.append(cleaned)

        # Recursively extract sub-ingredients.
        for group in parenthetical_groups:
            ingredients.extend(parse_level(group))

        return ingredients

    return parse_level(text)


### CONFIDENCE CALCULATION ###

def calculate_relevant_confidence(token_spans, relevant_ranges):
    """
    Identifies tokens within the ingredients and trace sections.
    These tokens are used for the confidence calculation.
    """

    relevant_tokens = []

    for token_start, token_end, token in token_spans:

        # Ignore punctuation-only tokens.
        if not any(char.isalnum() for char in token.word):
            continue

        # Check whether the token overlaps a relevant text section.
        for range_start, range_end in relevant_ranges:

            if token_start < range_end and token_end > range_start:
                relevant_tokens.append(token)
                break

    return relevant_tokens


### MAIN OCR FUNCTION ###

def run_ocr(image_bytes: bytes) -> OCRScanResult:
    """Preprocesses an image, performs OCR and extracts ingredients."""

    prep = preprocess_image(image_bytes)

    if not prep.success:
        return OCRScanResult(
            success=False,
            reason=prep.reason,
        )

    tokens = extract_tokens(prep.image)

    if not tokens:
        return OCRScanResult(
            success=False,
            reason="NO_TEXT_DETECTED",
        )

    # Reconstruct text and retain the positions of individual tokens.
    raw_text, token_spans = build_text_with_spans(tokens)

    # Extract ingredients, trace statements and their character ranges.
    ingredients_text, trace_statements, relevant_ranges = (
        parse_text_with_ranges(raw_text)
    )

    ingredients = split_ingredients(ingredients_text)

    # Calculate confidence using relevant ingredient/trace text.
    relevant_tokens = calculate_relevant_confidence(
        token_spans,
        relevant_ranges,
    )

    # If the ingredients heading was not recognised, or no relevant
    # tokens were identified, score all recognised alphanumeric words.
    if not ingredients_text or not relevant_tokens:
        relevant_tokens = [
            token for token in tokens
            if any(char.isalnum() for char in token.word)
        ]

    scored = [token.confidence for token in relevant_tokens]

    mean_confidence = (
        float(np.mean(scored)) if scored else 0.0
    )

    has_low_conf = any(
        token.is_low_conf for token in relevant_tokens
    )

    result = OCRScanResult(
        success=False,
        raw_text=raw_text,
        ingredients=ingredients,
        trace_statements=trace_statements,
        tokens=tokens,
        mean_confidence=mean_confidence,
        has_low_conf=has_low_conf,
    )

    # Reject scans with insufficient confidence.
    if mean_confidence < MEAN_CONF_THRESHOLD:

        result.reason = "LOW_CONFIDENCE"

    # Do not allow an empty ingredient list to produce a successful scan.
    elif not ingredients:

        result.reason = "NO_INGREDIENTS_FOUND"

    else:

        result.success = True

    return result


### TESTING ###

if __name__ == "__main__":

    try:
        with open(TEST_IMAGE, "rb") as f:
            result = run_ocr(f.read())

        print(f"Success: {result.success}")
        print(f"Reason: {result.reason}")
        print(f"Mean confidence: {result.mean_confidence:.1f}")
        print(f"Low confidence words: {result.has_low_conf}")
        print(f"Ingredients: {result.ingredients}")
        print(f"Trace statements: {result.trace_statements}")

        print("\nLow-confidence tokens:")

        for token in result.tokens:
            if token.is_low_conf:
                print(
                    f"{token.word}: {token.confidence:.1f}"
                )

    except FileNotFoundError:
        print(f"Could not find {TEST_IMAGE}")