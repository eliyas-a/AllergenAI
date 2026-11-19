from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

### CONSTANTS ###
TARGET_WIDTH = 1800          # every image is resized to this width (keeps scan time predictable, criterion #20)
SHARPNESS_THRESHOLD = 100.0  # minimum Laplacian variance, measured AFTER resizing so it is consistent
USE_SHARPENING = False       # experimental: switch on to compare confidence with/without
MEDIAN_KERNEL = 5            # must be odd; wider than the grain left after resizing
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID = (8, 8)
ADAPTIVE_BLOCK_SIZE = 51     # must be odd; larger than a character stroke at TARGET_WIDTH
ADAPTIVE_C = 12              # higher = thinner strokes, less flat-area speckle
MIN_COMPONENT_AREA = 15      # blobs smaller than this (in pixels) are treated as noise
ROI_MIN_AREA_RATIO = 0.10    # candidate text box must cover 10-90% of the image
ROI_MAX_AREA_RATIO = 0.90
ROI_PADDING = 10             # pixels kept around the detected box


@dataclass
class PreprocessResult:
    success: bool
    image: Optional[np.ndarray] = None
    reason: Optional[str] = None


def convert_to_greyscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image array into a 1-channel greyscale image."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_to_target_width(grey_image: np.ndarray) -> np.ndarray:
    """Scales the image to TARGET_WIDTH. INTER_AREA when shrinking (avoids aliasing),
    INTER_CUBIC when enlarging (keeps glyph shapes smooth)."""
    height, width = grey_image.shape[:2]
    scale = TARGET_WIDTH / width
    new_dim = (TARGET_WIDTH, int(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(grey_image, new_dim, interpolation=interpolation)


def calculate_sharpness(grey_image: np.ndarray) -> float:
    """Variance of the Laplacian. Low variance means few sharp edges, so the image is blurry."""
    return float(cv2.Laplacian(grey_image, cv2.CV_64F).var())


def remove_noise(grey_image: np.ndarray) -> np.ndarray:
    """Median blur removes salt-and-pepper grain while keeping edges
    (a Gaussian blur would smear them)."""
    return cv2.medianBlur(grey_image, MEDIAN_KERNEL)


def sharpen_image(grey_image: np.ndarray) -> np.ndarray:
    """Optional sharpening kernel. Off by default because it amplifies grain
    and thickens strokes before binarisation."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(grey_image, -1, kernel)


def normalize_contrast(grey_image: np.ndarray) -> np.ndarray:
    """Stretches pixel intensities across the full 0-255 range to fix dim lighting."""
    return cv2.normalize(grey_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def enhance_contrast(grey_image: np.ndarray) -> np.ndarray:
    """Applies CLAHE to even out contrast across shadows and glare."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(grey_image)


def apply_threshold(contrasted_image: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold: black text on white background."""
    return cv2.adaptiveThreshold(
        contrasted_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )


def remove_small_components(binary_image: np.ndarray) -> np.ndarray:
    """Deletes tiny black blobs (speckle) using connected-component analysis.
    Text is black on white, so the image is inverted first to make text the foreground."""
    inverted = cv2.bitwise_not(binary_image)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    cleaned = np.zeros_like(inverted)
    for label in range(1, count):  # label 0 is the background
        if stats[label, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            cleaned[labels == label] = 255
    return cv2.bitwise_not(cleaned)


def find_text_region(grey_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Looks for the largest four-sided contour (the boxed ingredients panel).
    Returns (x, y, w, h), or None if nothing suitable is found."""
    blurred = cv2.GaussianBlur(grey_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)  # closes gaps in the border

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = grey_image.shape[0] * grey_image.shape[1]

    best_box = None
    best_area = 0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        if ROI_MIN_AREA_RATIO * image_area <= area <= ROI_MAX_AREA_RATIO * image_area:
            if area > best_area:
                best_area = area
                best_box = (x, y, w, h)
    return best_box


def crop_to_region(grey_image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crops to the detected box with a small padding, clamped to the image edges."""
    x, y, w, h = box
    height, width = grey_image.shape[:2]
    x0, y0 = max(x - ROI_PADDING, 0), max(y - ROI_PADDING, 0)
    x1, y1 = min(x + w + ROI_PADDING, width), min(y + h + ROI_PADDING, height)
    return grey_image[y0:y1, x0:x1]


def preprocess_image(image_bytes: bytes, use_roi: bool = True) -> PreprocessResult:
    if not image_bytes:
        return PreprocessResult(success=False, reason="EMPTY_INPUT")

    try:
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            return PreprocessResult(success=False, reason="INVALID_IMAGE_DECODE")

        # 1. Greyscale, then fixed working size so thresholds behave the same on every phone
        grey = resize_to_target_width(convert_to_greyscale(image))

        # 2. Reject blurry images before spending time on the rest (criterion #8)
        if calculate_sharpness(grey) < SHARPNESS_THRESHOLD:
            return PreprocessResult(success=False, reason="BLURRY")

        # 3. Crop to the ingredients panel; fall back to the full image if none is found
        if use_roi:
            box = find_text_region(grey)
            if box is not None:
                grey = crop_to_region(grey, box)

        # 4. Denoise BEFORE anything that amplifies detail
        denoised = remove_noise(grey)
        if USE_SHARPENING:
            denoised = sharpen_image(denoised)

        # 5. Contrast: stretch range, then local equalisation
        contrasted = enhance_contrast(normalize_contrast(denoised))

        # 6. Binarise, then remove leftover speckle
        binary = apply_threshold(contrasted)
        cleaned = remove_small_components(binary)

        return PreprocessResult(success=True, image=cleaned)

    except Exception as e:
        return PreprocessResult(success=False, reason=f"PIPELINE_ERROR: {str(e)}")


def save_processed_image(processed_image: np.ndarray, output_path: str) -> bool:
    """Saves the processed image to a separate output file."""
    return cv2.imwrite(output_path, processed_image)


def measure_confidence(image: np.ndarray, lang: str = "eng", psm: int = 6) -> Tuple[float, int]:
    """Returns (mean word confidence, number of words under 70) for tuning evidence."""
    import pytesseract
    data = pytesseract.image_to_data(
        image, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    scores = [
        float(c) for c, w in zip(data["conf"], data["text"])
        if w.strip() and float(c) >= 0
    ]
    if not scores:
        return 0.0, 0
    return sum(scores) / len(scores), sum(1 for s in scores if s < 70)


if __name__ == "__main__":
    test_path = "processing_test_image.jpg"
    output_path = "processed_output.png"

    try:
        with open(test_path, "rb") as f:
            result = preprocess_image(f.read())
        print(f"Preprocessing Success: {result.success}")

        if result.success and result.image is not None:
            print(f"Processed Image Saved: {save_processed_image(result.image, output_path)}")
            print(f"Processed Image Shape: {result.image.shape}")
            mean_conf, low_words = measure_confidence(result.image)
            print(f"Mean confidence: {mean_conf:.1f} | Words under 70: {low_words}")
        else:
            print(f"Reason: {result.reason}")

    except FileNotFoundError:
        print(f"Add '{test_path}' to your folder to test preprocessing independently.")