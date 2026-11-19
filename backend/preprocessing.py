from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


### CONSTANTS ###

TARGET_WIDTH = 2000

MEDIAN_KERNEL = 3

CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID = (8, 8)

ADAPTIVE_BLOCK_SIZE = 51
ADAPTIVE_C = 12

MIN_COMPONENT_AREA = 12

MIN_ROI_RATIO = 0.05        # crop must cover at least 5% of the image, otherwise it is probably noise
ROI_PADDING = 25
MAX_LINE_RATIO = 0.20   # components wider or taller than this fraction of the image are borders, not text

### FILES ###
TEST_IMAGE = "processing_test_image.jpg"   # change the test image HERE ONLY
OUTPUT_IMAGE = "processed_output.png"


@dataclass
class PreprocessResult:
    success: bool
    image: Optional[np.ndarray] = None
    reason: Optional[str] = None
    roi_found: bool = False


def convert_to_greyscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image into greyscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def crop_to_label(grey_image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Crops to the densest block of text, ignoring borders and outlines."""

    height, width = grey_image.shape[:2]

    # Inverted threshold: dark text becomes white on a black background
    blurred = cv2.GaussianBlur(grey_image, (5, 5), 0)
    text_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    # Remove borders: letters are small, but outlines and panel edges
    # are connected shapes that span a large part of the image
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_mask,
        connectivity=8
    )
    too_wide = stats[:, cv2.CC_STAT_WIDTH] > MAX_LINE_RATIO * width
    too_tall = stats[:, cv2.CC_STAT_HEIGHT] > MAX_LINE_RATIO * height
    remove = too_wide | too_tall
    remove[0] = False                     # label 0 is the background
    text_mask[remove[labels]] = 0

    # Merge letters into words and lines into a paragraph block.
    # The kernel is taller than the gap between lines but narrower than
    # the gap between the ingredients and the nutrition panel.
    kernel_w = max(width // 60, 3)
    kernel_h = max(width // 50, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    merged = cv2.dilate(text_mask, kernel, iterations=1)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        merged,
        connectivity=8
    )

    if count <= 1:
        return grey_image, False

    # Largest blob by pixel count (label 0 is the background)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    x = stats[largest, cv2.CC_STAT_LEFT]
    y = stats[largest, cv2.CC_STAT_TOP]
    box_w = stats[largest, cv2.CC_STAT_WIDTH]
    box_h = stats[largest, cv2.CC_STAT_HEIGHT]

    # If the box is tiny, it is probably not a real label
    if box_w * box_h < MIN_ROI_RATIO * width * height:
        return grey_image, False

    x0 = max(x - ROI_PADDING, 0)
    y0 = max(y - ROI_PADDING, 0)
    x1 = min(x + box_w + ROI_PADDING, width)
    y1 = min(y + box_h + ROI_PADDING, height)

    return grey_image[y0:y1, x0:x1], True

def resize_to_target(
    grey_image: np.ndarray,
    target_width: int = TARGET_WIDTH
) -> np.ndarray:
    """Resizes the image to a consistent width."""

    height, width = grey_image.shape[:2]

    scale = target_width / width
    new_dim = (target_width, int(height * scale))

    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC

    return cv2.resize(
        grey_image,
        new_dim,
        interpolation=method
    )


def remove_noise(grey_image: np.ndarray) -> np.ndarray:
    """Uses median filtering to reduce image noise."""

    return cv2.medianBlur(grey_image, MEDIAN_KERNEL)


def normalize_contrast(grey_image: np.ndarray) -> np.ndarray:
    """Normalises pixel intensities across the 0-255 range."""

    return cv2.normalize(
        grey_image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )


def enhance_contrast(grey_image: np.ndarray) -> np.ndarray:
    """Applies CLAHE to improve local contrast."""

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )

    return clahe.apply(grey_image)


def apply_threshold(contrasted_image: np.ndarray) -> np.ndarray:
    """Converts the image into a binary black-and-white image."""

    return cv2.adaptiveThreshold(
        contrasted_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )


def remove_specks(binary_image: np.ndarray) -> np.ndarray:
    """Removes small isolated connected components."""

    inverted = cv2.bitwise_not(binary_image)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted,
        connectivity=8
    )

    cleaned = np.zeros_like(inverted)

    for label in range(1, count):

        area = stats[label, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == label] = 255

    return cv2.bitwise_not(cleaned)


def preprocess_image(image_bytes: bytes, use_crop: bool = True) -> PreprocessResult:

    if not image_bytes:
        return PreprocessResult(
            success=False,
            reason="EMPTY_INPUT"
        )

    try:

        np_array = np.frombuffer(image_bytes, np.uint8)

        image = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return PreprocessResult(
                success=False,
                reason="INVALID_IMAGE_DECODE"
            )

        # 1. Convert to greyscale
        grey = convert_to_greyscale(image)

        # 2. Detect and crop the region of interest (can be switched off)
        if use_crop:
            cropped, roi_found = crop_to_label(grey)
        else:
            cropped, roi_found = grey, False

        # 3. Resize to a consistent width
        resized = resize_to_target(cropped)

        # 4. Reduce image noise
        denoised = remove_noise(resized)

        # 5. Normalise and enhance contrast
        normalised = normalize_contrast(denoised)
        contrasted = enhance_contrast(normalised)

        # 6. Apply adaptive thresholding
        binary = apply_threshold(contrasted)

        # 7. Remove small noise components
        cleaned = remove_specks(binary)

        return PreprocessResult(
            success=True,
            image=cleaned,
            roi_found=roi_found
        )

    except Exception as e:

        return PreprocessResult(
            success=False,
            reason=f"PIPELINE_ERROR: {str(e)}"
        )


def save_processed_image(
    processed_image: np.ndarray,
    output_path: str
) -> bool:
    """Saves the processed image to a file."""

    return cv2.imwrite(output_path, processed_image)


if __name__ == "__main__":

    try:

        with open(TEST_IMAGE, "rb") as f:

            result = preprocess_image(f.read())

            print(f"Preprocessing Success: {result.success}")

            if result.success and result.image is not None:

                print(f"ROI Found: {result.roi_found}")

                saved = save_processed_image(result.image, OUTPUT_IMAGE)

                print(f"Processed Image Saved: {saved}")
                print(f"Processed Image Shape: {result.image.shape}")

            else:
                print(f"Reason: {result.reason}")

    except FileNotFoundError:

        print(f"Could not find {TEST_IMAGE}")