from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

### CONSTANTS ###
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
ADAPTIVE_BLOCK_SIZE = 15  
ADAPTIVE_C = 12
UPSCALE_FACTOR = 1.5

@dataclass
class PreprocessResult:
    success: bool
    image: Optional[np.ndarray] = None
    reason: Optional[str] = None

def convert_to_greyscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image array into a 1-channel greyscale image."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def upscale_image(grey_image: np.ndarray, factor: float = UPSCALE_FACTOR) -> np.ndarray:
    """Resizes image to increase DPI, helping separate dense text strokes."""
    height, width = grey_image.shape[:2]
    new_dim = (int(width * factor), int(height * factor))
    return cv2.resize(
        grey_image, new_dim, interpolation=cv2.INTER_CUBIC
        )

def normalize_contrast(grey_image: np.ndarray) -> np.ndarray:
    """Stretches pixel intensities across the full 0-255 spectrum to fix dim lighting."""
    return cv2.normalize(
        grey_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )

def enhance_contrast(grey_image: np.ndarray) -> np.ndarray:
    """Applies CLAHE to equalise contrast."""
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID
        )
    return clahe.apply(grey_image)

def remove_noise(grey_image: np.ndarray) -> np.ndarray:
    """Applies Gaussian blur to smooth camera grain."""
    return cv2.GaussianBlur(grey_image, (3, 3), 0)

def sharpen_image(grey_image: np.ndarray) -> np.ndarray:
    """Applies a sharpening matrix to accentuate text edges before binarisation."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(grey_image, -1, kernel)

def apply_threshold(blurred_image: np.ndarray) -> np.ndarray:
    """Applies adaptive Gaussian thresholding to binarise the image."""
    thresh = cv2.adaptiveThreshold(
        blurred_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )
    return thresh

def preprocess_image(image_bytes: bytes) -> PreprocessResult:
    if not image_bytes:
        return PreprocessResult(success=False, reason="EMPTY_INPUT")

    try:
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return PreprocessResult(success=False, reason="INVALID_IMAGE_DECODE")

        # 1. Convert to grayscale
        grey = convert_to_greyscale(image)
        
        # 2. Upscale image to improve text resolution (DPI)
        upscaled = upscale_image(grey)

        sharp = sharpen_image(upscaled)

        # 3. Light noise reduction BEFORE contrast boost
        blurred = remove_noise(sharp)

        # 4. Stretch lighting range & apply CLAHE
        norm = normalize_contrast(blurred)
        contrasted = enhance_contrast(norm)

        # 5. Fine-tuned adaptive binarisation with higher C
        thresh = apply_threshold(contrasted)

        return PreprocessResult(success=True, image=thresh)

    except Exception as e:
        return PreprocessResult(success=False, reason=f"PIPELINE_ERROR: {str(e)}")


def save_processed_image(
        processed_image: np.ndarray, output_path: str
        ) -> bool:
    """Saves the processed image to a separate output file."""
    return cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    # Test script for preprocessing module
    test_path = "test_image.jpg"
    output_path = "processed_output.png"

    try:
        with open(test_path, "rb") as f:
            result = preprocess_image(f.read())
            print(f"Preprocessing Success: {result.success}")

            if result.success and result.image is not None:
                saved = save_processed_image(result.image, output_path)
                print(f"Processed Image Saved: {saved}")
                print(f"Output File Path: {output_path}")
                print(f"Processed Image Shape: {result.image.shape}")
            else:
                print(f"Reason: {result.reason}")

    except FileNotFoundError:
        print(
            f"Add '{test_path}' to your folder to test preprocessing independently."
        )