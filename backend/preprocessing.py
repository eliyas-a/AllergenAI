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

MIN_ROI_RATIO = 0.15
MAX_ROI_RATIO = 0.90
ROI_PADDING = 10


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
    """Detects and crops the largest suitable rectangular region."""

    height, width = grey_image.shape[:2]

    blurred = cv2.GaussianBlur(grey_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_box = None
    best_area = 0

    for contour in contours:

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        if len(approx) != 4:
            continue

        x, y, box_w, box_h = cv2.boundingRect(approx)

        area = box_w * box_h
        ratio = area / (width * height)

        if MIN_ROI_RATIO <= ratio <= MAX_ROI_RATIO:
            if area > best_area:
                best_area = area
                best_box = (x, y, box_w, box_h)

    if best_box is None:
        return grey_image, False

    x, y, box_w, box_h = best_box

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


def preprocess_image(image_bytes: bytes) -> PreprocessResult:

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

        # 2. Detect and crop the region of interest
        cropped, roi_found = crop_to_label(grey)

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

    test_path = "processing_test_image.jpg"
    output_path = "processed_output.png"

    try:

        with open(test_path, "rb") as f:

            result = preprocess_image(f.read())

            print(f"Preprocessing Success: {result.success}")

            if result.success and result.image is not None:

                print(f"ROI Found: {result.roi_found}")

                saved = save_processed_image(
                    result.image,
                    output_path
                )

                print(f"Processed Image Saved: {saved}")
                print(f"Processed Image Shape: {result.image.shape}")

            else:
                print(f"Reason: {result.reason}")

    except FileNotFoundError:

        print(
            f"Add '{test_path}' to your folder "
            "to test preprocessing independently."
        )

