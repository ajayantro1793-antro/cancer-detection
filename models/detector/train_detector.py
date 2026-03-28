import sys
import yaml
from pathlib import Path
from ultralytics import YOLO

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# ── Load config ────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)
D = cfg["detector"]
P = cfg["paths"]

CLASS_NAMES = ["adenocarcinoma", "large_cell_carcinoma", "squamous_cell_carcinoma"]
def train_detector():

    # Verify dataset YAML exists (convert_to_yolo.py must be run first)
    dataset_yaml = Path(P["yolo_dataset"]) / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {dataset_yaml}.\n"
            f"Run: python models/detector/convert_to_yolo.py"
        )

    # Sanity-check: confirm the YAML has 3 classes, not 1
    with open(dataset_yaml) as f:
        ds_cfg = yaml.safe_load(f)
    if ds_cfg.get("nc") != 3:
        raise ValueError(
            f"Expected nc=3 in dataset.yaml (adenocarcinoma, large_cell_carcinoma, "
            f"squamous_cell_carcinoma), but got nc={ds_cfg.get('nc')}.\n"
            f"Re-run convert_to_yolo.py to regenerate the dataset."
        )

    print(f"Dataset      : {dataset_yaml}")
    print(f"Classes (3)  : {CLASS_NAMES}")
    print(f"Model        : {D['model']}")
    print(f"Image size   : {D['image_size']}")
    print(f"Batch size   : {D['batch_size']}")
    print(f"Epochs       : {D['epochs']}")

    # ── COMMENT OUT — model already trained, skip to avoid retraining ──
    # model = YOLO(D["model"])

    # results = model.train(
    #     data      = str(dataset_yaml),
    #     epochs    = D["epochs"],
    #     imgsz     = D["image_size"],
    #     batch     = D["batch_size"],
    #     lr0       = D["learning_rate"],
    #     iou       = D["iou_threshold"],
    #     conf      = D["confidence_threshold"],
    #     amp       = D["mixed_precision"],
    #     patience  = 15,
    #     save      = True,
    #     save_period = 5,
    #     project   = P["detector_output"],
    #     name      = "runs",
    #     exist_ok  = True,
    #     verbose   = True,
    #     flipud    = 0.3,
    #     fliplr    = 0.5,
    #     degrees   = 10.0,
    #     translate = 0.1,
    #     scale     = 0.2,
    #     mosaic    = 0.3,
    #     hsv_h     = 0.0,
    #     hsv_s     = 0.2,
    #     hsv_v     = 0.3,
    # )
    # ───────────────────────────────────────────────────────────────────

    print(f"\nSkipping training — loading existing best weights.")

    # ── FIXED PATH — points to your actual saved best.pt ───────────────
    best_weights = r"saved_models/yolov8_nodule_best.pt"

    # ── COMMENT OUT old wrong path ──────────────────────────────────────
    # best_weights = r"runs\detect\models\detector\runs\weights\best.pt"
    # ───────────────────────────────────────────────────────────────────

    print(f"Best weights : {best_weights}")
    # print(f"Results dir  : {results.save_dir}")   ← comment this too, results no longer exists

    # ── Validate on test set ─────────────────────────────────────────────
    print("\nRunning validation on test set...")
    best_model = YOLO(best_weights)
    metrics = best_model.val(
        data   = str(dataset_yaml),
        split  = "test",
        imgsz  = D["image_size"],
        conf   = D["confidence_threshold"],
        iou    = D["iou_threshold"],
    )

    print(f"\nTest Set Metrics:")
    print(f"  mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")

    print(f"\n  Per-class AP@0.5:")
    for i, cls_name in enumerate(CLASS_NAMES):
        try:
            ap = metrics.box.ap50[i]
            print(f"    [{i}] {cls_name:<30} : {ap:.4f}")
        except (IndexError, AttributeError):
            print(f"    [{i}] {cls_name:<30} : N/A")

    return metrics


if __name__ == "__main__":
    train_detector()

