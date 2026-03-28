"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
Preprocessing utilities for CT scan images.

Key ideas:
  - CT images are stored as grayscale with Hounsfield Unit (HU) values.
    HU values represent tissue density: air = -1000, water = 0, bone = +1000.
    We simulate a "soft-tissue window" by clipping to [-1000, 400] and
    normalising to [0, 255] — this enhances contrast for lung tissue.
  - We convert to 3-channel RGB because EfficientNet expects RGB input
    (it was pretrained on ImageNet which is RGB). The three channels will
    be identical but the model's first conv layer weights are designed for 3ch.
  - Lung masks from ct_lung_finder are used to zero-out non-lung regions,
    so the model focuses only on relevant tissue rather than ribs or fat.
─────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
from pathlib import Path


def load_and_preprocess(image_path: str, target_size: int = 512) -> np.ndarray:
    """
    Full preprocessing pipeline for a single CT slice.
    Returns a uint8 RGB numpy array of shape (target_size, target_size, 3).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    # Simulate HU windowing for soft tissue — enhances lung nodule visibility
    img = apply_hu_windowing(img)

    # Resize to model input size
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Convert to 3-channel RGB for EfficientNet compatibility
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def apply_hu_windowing(img: np.ndarray,
                        window_center: int = -600,
                        window_width: int = 1500) -> np.ndarray:
    """
    Apply a lung window to the CT image.
    Window center=-600 and width=1500 is the standard lung window used in
    radiology — it optimally shows lung parenchyma and nodules.

    The formula:  lower = center - width/2
                  upper = center + width/2
    Then clip and normalise to [0, 255].
    """
    lower = window_center - window_width // 2
    upper = window_center + window_width // 2

    # Clip pixel values to the window range
    img = np.clip(img.astype(np.float32), lower, upper)

    # Normalise to [0, 255]
    img = ((img - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return img


def apply_lung_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary lung segmentation mask to the image.
    Pixels outside the lung region are zeroed out — this reduces noise
    from ribs, spine, and surrounding tissue, helping the model concentrate
    on what actually matters.

    Args:
        image: HxWx3 RGB numpy array
        mask:  HxW or HxWx1 binary mask (0=background, 255=lung)
    Returns:
        Masked image of the same shape as input
    """
    # Ensure mask matches image dimensions
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # Binarise: anything above 127 is considered lung
    binary_mask = (mask > 127).astype(np.uint8)

    # Apply mask to each colour channel independently
    masked = image.copy()
    for channel in range(image.shape[2]):
        masked[:, :, channel] = image[:, :, channel] * binary_mask

    return masked


def normalise_for_model(img: np.ndarray) -> np.ndarray:
    """
    Normalise a uint8 [0,255] image to float32 using ImageNet statistics.
    This is required because EfficientNet-B4 was pretrained on ImageNet,
    so during fine-tuning the input distribution should match pretraining.

    ImageNet mean = [0.485, 0.456, 0.406]  (per channel)
    ImageNet std  = [0.229, 0.224, 0.225]
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = img.astype(np.float32) / 255.0   # scale to [0, 1]
    img = (img - mean) / std               # standardise
    return img


def resize_with_padding(img: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize an image to target_size x target_size while preserving aspect ratio
    by padding with zeros. This is an alternative to simple resize when you
    want to avoid distorting nodule shapes.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create blank canvas and centre the resized image
    canvas = np.zeros((target_size, target_size, img.shape[2]), dtype=img.dtype)
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

    return canvas


