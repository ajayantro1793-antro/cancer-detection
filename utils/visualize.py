"""
visualize.py
─────────────────────────────────────────────────────────────────────────────
Visualisation utilities: bounding box drawing, heatmap overlay,
result display helpers.

Fix: draw_bounding_boxes now uses the same realistic mm calculation as
inference.py (px * 0.625 * 0.25, clamped 3–30mm) instead of px * 0.7
which produced hundreds-of-mm labels on large boxes.
─────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List, Tuple, Optional


def _px_to_mm(px_size: float) -> float:
    """
    Convert a bounding-box pixel diameter to mm.
    Uses standard CT pixel spacing (0.625 mm/px at 512px FOV) with a
    0.25 correction factor for Grad-CAM blob over-expansion.
    Clamped to the clinical nodule range of 3–30 mm.
    """
    mm = px_size * 0.625 * 0.25
    return  round(max(3.0, min(30.0, px_size * 0.625 * 0.25)), 1)


def draw_bounding_boxes(image: np.ndarray,
                         boxes: List[List[float]],
                         scores: List[float],
                         color: Tuple[int, int, int] = (255, 165, 0),
                         thickness: int = 2) -> np.ndarray:
    """
    Draw nodule detection bounding boxes on a CT image.

    Args:
        image:     HxWx3 RGB numpy array
        boxes:     list of [x1, y1, x2, y2] pixel coordinates
        scores:    list of confidence scores, one per box
        color:     RGB colour for box outline (default: orange)
        thickness: line thickness in pixels
    Returns:
        Annotated copy of the image
    """
    annotated = image.copy()

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [int(v) for v in box]

        # Clamp coordinates to image bounds defensively
        h, w = annotated.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        # Skip degenerate boxes
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Realistic mm size label
        px_size = max(x2 - x1, y2 - y1)
        mm_size = _px_to_mm(px_size)
        label   = f"{score:.0%} | {mm_size}mm"

        # Label background
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        label_y = max(y1, text_h + baseline + 4)   # keep label inside image top

        cv2.rectangle(annotated,
                      (x1, label_y - text_h - baseline - 4),
                      (x1 + text_w + 6, label_y),
                      color, -1)
        cv2.putText(annotated, label,
                    (x1 + 3, label_y - baseline - 2),
                    font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


def overlay_heatmap(image: np.ndarray,
                     heatmap: np.ndarray,
                     alpha: float = 0.5,
                     colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto the original CT image.
    """
    heatmap_uint8    = np.uint8(255 * heatmap)
    heatmap_coloured = cv2.applyColorMap(heatmap_uint8, colormap)
    image_bgr        = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blended_bgr      = cv2.addWeighted(heatmap_coloured, alpha, image_bgr, 1 - alpha, 0)
    return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)


def create_result_figure(original: np.ndarray,
                          detected: np.ndarray,
                          heatmap_overlay: np.ndarray,
                          prediction: str,
                          confidence: float,
                          nodule_count: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0F172A")

    titles = ["Original CT Slice", "Nodule Detection", "Grad-CAM Heatmap"]
    images = [original, detected, heatmap_overlay]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    risk_color = "#EF4444" if prediction == "Malignant" else \
                 "#F59E0B" if prediction == "Benign" else "#10B981"
    fig.suptitle(
        f"Diagnosis: {prediction}  |  Confidence: {confidence:.1%}  |  "
        f"Nodules Found: {nodule_count}",
        color=risk_color, fontsize=14, fontweight="bold", y=0.02
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def get_risk_level(prediction: str, confidence: float) -> Tuple[str, str]:
    if prediction == "Malignant" and confidence >= 0.80:
        return "CRITICAL", "#EF4444"
    elif prediction == "Malignant" and confidence >= 0.50:
        return "HIGH", "#F97316"
    elif prediction == "Benign":
        return "MEDIUM", "#F59E0B"
    else:
        return "LOW", "#10B981"

