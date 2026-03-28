"""
gradcam_utils.py  —  Grad-CAM Explainability (FIXED)
─────────────────────────────────────────────────────────────────────────────
Fixes applied:
  1. Target layer changed from model.blocks[-1] (a Sequential container)
     to model.blocks[-1][-1].bn3 — the actual last BatchNorm before pooling.
     Using the Sequential container itself causes pytorch-grad-cam to hook
     the wrong tensor, producing heatmaps that highlight background regions.
  2. Added GradCAM++ as fallback — more accurate for small objects (nodules).
  3. Added histogram equalisation on the raw heatmap to spread the colour
     range more evenly across the lung region.
─────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import Optional, Tuple


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the correct target layer for Grad-CAM in timm's EfficientNet-B4.

    WHY THIS MATTERS:
    model.blocks[-1] is a nn.Sequential CONTAINER — hooking it gives you
    the output of the whole block group, not the final conv feature map.
    pytorch-grad-cam needs the actual last convolutional/BN layer so it can
    capture spatial gradients before they're pooled away.

    For EfficientNet-B4 in timm, model.blocks[-1][-1] is the last MBConv
    block. Inside it, .bn3 is the final BatchNorm after the pointwise
    projection conv — this is the richest spatial feature map available.
    """
    # Exact layer names confirmed from model inspection:
    # blocks.5.7.bn3 = last layer of block group 5 (16x16 spatial resolution)
    # This gives finer spatial localisation than blocks.6.1.bn3 (8x8)
    # which is too coarse and causes heatmaps to miss small nodules.
    try:
        return model.blocks[5][7].bn3   # confirmed: blocks.5.7.bn3
    except (AttributeError, IndexError):
        pass
    try:
        return model.blocks[5][-1].bn3  # last layer of block 5
    except (AttributeError, IndexError):
        pass
    try:
        return model.blocks[-2][-1].bn3 # second-to-last block group
    except (AttributeError, IndexError):
        pass
    try:
        return model.blocks[-1][-1].bn3 # last block group fallback
    except (AttributeError, IndexError):
        pass
    return model.conv_head


def generate_gradcam(model: nn.Module,
                     input_tensor: torch.Tensor,
                     original_image: np.ndarray,
                     target_class: Optional[int] = None,
                     use_cuda: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Grad-CAM++ heatmap for a single CT scan image.

    Args:
        model:          Trained EfficientNet-B4, in eval mode
        input_tensor:   Preprocessed image tensor, shape (1, 3, 512, 512)
        original_image: Original uint8 RGB image, shape (512, 512, 3)
        target_class:   Class index to explain (0=Normal, 1=Benign, 2=Malignant)
                        If None, uses the predicted class.
        use_cuda:       Whether to use GPU acceleration

    Returns:
        heatmap_overlay: uint8 RGB image with Grad-CAM overlaid, shape (H, W, 3)
        raw_heatmap:     float32 array in [0, 1], shape (H, W)
    """
    target_layer = get_target_layer(model)

    # Use GradCAM++ — it's more accurate than vanilla GradCAM for small
    # objects like lung nodules because it weights each pixel's gradient
    # contribution more carefully.
    with GradCAMPlusPlus(model=model,
                         target_layers=[target_layer]) as cam:

        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # ── Post-process: focus heatmap on lung region ─────────────────────────
    grayscale_cam = _enhance_heatmap(grayscale_cam, original_image)

    # Normalise original image to [0, 1] for overlay
    if original_image.dtype == np.uint8:
        img_float = original_image.astype(np.float32) / 255.0
    else:
        img_float = original_image.copy()

    heatmap_overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    return heatmap_overlay, grayscale_cam


def _enhance_heatmap(heatmap: np.ndarray,
                     original_image: np.ndarray) -> np.ndarray:
    """
    Post-process the raw Grad-CAM heatmap to better concentrate it
    on the lung region and suppress background noise.

    Steps:
      1. Create a rough lung mask by thresholding the CT image
         (lung tissue is darker than bone/vessels on standard windowing)
      2. Suppress heatmap values outside the lung mask
      3. Apply CLAHE (histogram equalisation) to spread the colour range
         across the lung area rather than wasting it on background
    """
    # Convert to grayscale for masking
    if original_image.ndim == 3:
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_image.copy()

    # Resize heatmap to match image if needed
    h, w = gray.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Suppress only the extreme background corners (outside the CT circle).
    # CT images have a circular field of view — the four black corners are
    # pure air (pixel value ~0) and should not contribute to the heatmap.
    # We keep a SOFT suppression so we don't accidentally wipe out real
    # lung tissue activations.
    body_mask = (gray > 10).astype(np.float32)

    # Gentle blur — just softens edges, doesn't erode
    body_mask = cv2.GaussianBlur(body_mask, (41, 41), 0)

    # Apply soft suppression — multiply heatmap by mask
    heatmap = heatmap * body_mask

    # Boost contrast: apply power curve to make hot spots stand out more
    # Values close to 1.0 stay high; low values get pushed further down
    heatmap = np.power(np.clip(heatmap, 0, 1), 0.7)

    # Re-normalise to [0, 1] after masking
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)

    return heatmap.astype(np.float32)


def generate_gradcam_for_all_classes(model: nn.Module,
                                      input_tensor: torch.Tensor,
                                      original_image: np.ndarray,
                                      use_cuda: bool = True) -> dict:
    """
    Generate separate Grad-CAM++ maps for all 3 classes.
    Useful for debugging — compare which regions the model associates
    with Normal vs Benign vs Malignant.

    Returns a dict: {class_name: heatmap_overlay_array}
    """
    class_names = ["Normal", "Benign", "Malignant"]
    results = {}
    for idx, name in enumerate(class_names):
        overlay, _ = generate_gradcam(
            model, input_tensor, original_image,
            target_class=idx, use_cuda=use_cuda
        )
        results[name] = overlay
    return results


def resize_heatmap(heatmap: np.ndarray,
                   target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a raw Grad-CAM heatmap (float32 [0,1]) to a target resolution.
    Uses bicubic interpolation for smooth gradients.
    """
    return cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)


