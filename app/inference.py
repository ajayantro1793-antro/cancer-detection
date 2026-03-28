"""
inference.py  —  Model Loading & Prediction Pipeline
"""

import cv2
import yaml
import torch
import numpy as np
import timm
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from gradcam.gradcam_utils import generate_gradcam
from utils.preprocess import load_and_preprocess, normalise_for_model
from utils.visualize import draw_bounding_boxes, get_risk_level
from models.classifier.dataset import CLASS_NAMES

with open(Path(__file__).resolve().parents[1] / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

C      = cfg["classifier"]
D      = cfg["detector"]
P      = cfg["paths"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier(weights_path: str) -> torch.nn.Module:
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=False,
        num_classes=C["num_classes"],
        drop_rate=C["dropout"]
    )
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    print(f"Classifier loaded from {weights_path}")
    print(f"  Best val accuracy: {checkpoint.get('val_acc', 'N/A'):.3f}")
    return model


def load_detector(weights_path: str) -> YOLO:
    model = YOLO(weights_path)
    print(f"Detector loaded from {weights_path}")
    return model


def _px_to_mm(px_size: float) -> float:
    mm = px_size * 0.625 * 0.25
    return round(max(3.0, min(30.0, mm)), 1)


def find_peak_boxes(grayscale_cam, image_size, prediction, confidence,
                    box_half=28, suppress_radius=80, max_peaks=3, min_activation=0.50):
    """Find top-N hotspot peaks using iterative argmax + suppression.
    Each box is exactly (box_half*2) x (box_half*2) pixels — always small."""
    if prediction == "Normal":
        return [], []

    cam = grayscale_cam.copy().astype(np.float32)
    boxes, scores = [], []

    for _ in range(max_peaks):
        peak_val = cam.max()
        if peak_val < min_activation:
            break

        peak_idx = np.unravel_index(np.argmax(cam), cam.shape)
        row, col = int(peak_idx[0]), int(peak_idx[1])

        x1 = max(0,          col - box_half)
        y1 = max(0,          row - box_half)
        x2 = min(image_size, col + box_half)
        y2 = min(image_size, row + box_half)

        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
            box_score = round(float(np.clip(peak_val * confidence, 0.30, 0.92)), 3)
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            scores.append(box_score)

        # Zero out neighbourhood so next iteration finds a different peak
        r1 = max(0,          row - suppress_radius)
        r2 = min(image_size, row + suppress_radius)
        c1 = max(0,          col - suppress_radius)
        c2 = min(image_size, col + suppress_radius)
        cam[r1:r2, c1:c2] = 0.0

    return boxes, scores


def predict(image_input, classifier, detector, conf_threshold=0.25):

    # ── Step 1: Load image ────────────────────────────────────────────────
    if isinstance(image_input, (str, Path)):
        original = load_and_preprocess(str(image_input), target_size=C["image_size"])
    elif isinstance(image_input, Image.Image):
        arr      = np.array(image_input.convert("RGB"))
        original = cv2.resize(arr, (C["image_size"], C["image_size"]))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        original = cv2.resize(image_input, (C["image_size"], C["image_size"]))
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")

    # ── Step 2: Classification ────────────────────────────────────────────
    normalised   = normalise_for_model(original)
    input_tensor = torch.from_numpy(normalised).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            logits = classifier(input_tensor)
            probs  = torch.softmax(logits, dim=1)[0]

    probs_np      = probs.cpu().numpy()
    pred_idx      = int(np.argmax(probs_np))
    prediction    = CLASS_NAMES[pred_idx]
    confidence    = float(probs_np[pred_idx])
    probabilities = {name: float(p) for name, p in zip(CLASS_NAMES, probs_np)}
    risk_level, risk_color = get_risk_level(prediction, confidence)

    # ── Step 3: Grad-CAM ──────────────────────────────────────────────────
    heatmap_image, grayscale_cam = generate_gradcam(
        model          = classifier,
        input_tensor   = input_tensor,
        original_image = original,
        target_class   = pred_idx,
        use_cuda       = (DEVICE.type == "cuda")
    )

    # ── Step 4: YOLO — filter out full-image false boxes ──────────────────
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    det_results  = detector.predict(
        source=original_bgr,
        imgsz=D["image_size"],
        conf=conf_threshold,
        iou=D["iou_threshold"],
        verbose=False
    )[0]

    boxes, scores = [], []
    max_allowed   = C["image_size"] * 0.25   # reject boxes wider/taller than 25% of image

    if det_results.boxes is not None and len(det_results.boxes) > 0:
        for box, score in zip(det_results.boxes.xyxy.cpu().numpy(),
                               det_results.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            print(f"YOLO box: x1={x1:.0f} y1={y1:.0f} x2={x2:.0f} y2={y2:.0f} w={w:.0f} h={h:.0f} score={score:.2f}")
            if w <= max_allowed and h <= max_allowed:
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                scores.append(float(score))

    print(f"YOLO boxes after size filter: {len(boxes)}")

    # ── Step 5: Fallback to Grad-CAM peaks if no valid YOLO boxes ─────────
    used_fallback = False
    if len(boxes) == 0:
        boxes, scores = find_peak_boxes(
            grayscale_cam   = grayscale_cam,
            image_size      = C["image_size"],
            prediction      = prediction,
            confidence      = confidence,
            box_half        = 28,
            suppress_radius = 80,
            max_peaks       = 3,
            min_activation  = 0.50,
        )
        used_fallback = True
        print(f"Fallback boxes: {boxes}")

    # ── Step 6: Nodule sizes ──────────────────────────────────────────────
    nodule_sizes_mm = [_px_to_mm(max(b[2]-b[0], b[3]-b[1])) for b in boxes]

    detection_image = draw_bounding_boxes(original, boxes, scores)

    return {
        "prediction":      prediction,
        "confidence":      confidence,
        "probabilities":   probabilities,
        "risk_level":      risk_level,
        "risk_color":      risk_color,
        "boxes":           boxes,
        "scores":          scores,
        "nodule_count":    len(boxes),
        "nodule_sizes_mm": nodule_sizes_mm,
        "original_image":  original,
        "detection_image": detection_image,
        "heatmap_image":   heatmap_image,
        "used_fallback":   used_fallback,
    }

