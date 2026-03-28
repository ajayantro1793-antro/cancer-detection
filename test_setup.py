"""
test_setup.py  —  Pre-Hackathon System Check
─────────────────────────────────────────────────────────────────────────────
Run this FIRST to verify your entire environment is correctly configured
before the hackathon starts. It checks:

  ✓ Python version
  ✓ CUDA availability and GPU details
  ✓ All required packages installed
  ✓ VRAM is sufficient for training
  ✓ Dataset directories exist
  ✓ Config file is valid
  ✓ A dummy forward pass through both model architectures
  ✓ Mixed precision works correctly
  ✓ Streamlit can be imported

Usage:
    python test_setup.py

All checks should print ✓. Fix any ✗ before starting.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

results = []

def check(label: str, passed: bool, detail: str = "", warn_only: bool = False):
    symbol = PASS if passed else (WARN if warn_only else FAIL)
    line   = f"  {symbol}  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    results.append(passed or warn_only)
    return passed


print("=" * 60)
print("  LUNG CANCER DETECTION — SYSTEM CHECK")
print("=" * 60)

# ── 1. Python Version ──────────────────────────────────────────────────────
print("\n[ Environment ]")
v = sys.version_info
check("Python version",
      v.major == 3 and v.minor >= 10,
      f"Python {v.major}.{v.minor}.{v.micro}  (need 3.10+)")

# ── 2. CUDA & GPU ──────────────────────────────────────────────────────────
print("\n[ GPU & CUDA ]")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    check("CUDA available", cuda_ok,
          f"CUDA {torch.version.cuda}" if cuda_ok else "CPU only — training will be slow")

    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("GPU detected",    True,  gpu_name)
        check("VRAM sufficient", vram_gb >= 4,
              f"{vram_gb:.1f} GB  (need ≥4GB, 6GB recommended)")

        # Test mixed precision
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            x = torch.randn(2, 3, 64, 64).cuda()
            with autocast():
                y = x * 2
            check("Mixed precision (AMP)", True, "float16 OK")
        except Exception as e:
            check("Mixed precision (AMP)", False, str(e))
    else:
        check("GPU detected",    False, "No GPU found", warn_only=True)
        check("VRAM sufficient", False, "No GPU",       warn_only=True)

except ImportError:
    check("PyTorch installed", False, "pip install torch")

# ── 3. Required Packages ───────────────────────────────────────────────────
print("\n[ Required Packages ]")
packages = [
    ("torch",           "PyTorch"),
    ("torchvision",     "torchvision"),
    ("timm",            "timm (EfficientNet)"),
    ("ultralytics",     "Ultralytics (YOLOv8)"),
    ("streamlit",       "Streamlit"),
    ("cv2",             "OpenCV"),
    ("PIL",             "Pillow"),
    ("numpy",           "NumPy"),
    ("pandas",          "Pandas"),
    ("sklearn",         "scikit-learn"),
    ("albumentations",  "Albumentations"),
    ("pytorch_grad_cam","pytorch-grad-cam"),
    ("fpdf",            "fpdf2"),
    ("matplotlib",      "Matplotlib"),
    ("seaborn",         "Seaborn"),
    ("yaml",            "PyYAML"),
    ("tqdm",            "tqdm"),
]

for module, name in packages:
    try:
        pkg = __import__(module)
        ver = getattr(pkg, "__version__", "?")
        check(name, True, f"v{ver}")
    except ImportError:
        check(name, False, f"pip install {name.lower().replace(' ','')}")

# ── 4. Config File ─────────────────────────────────────────────────────────
print("\n[ Config ]")
try:
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    required_keys = ["classifier", "detector", "paths", "augmentation"]
    all_keys_ok = all(k in cfg for k in required_keys)
    check("config.yaml valid", all_keys_ok,
          "All required sections present" if all_keys_ok else "Missing sections")
except FileNotFoundError:
    check("config.yaml exists", False, "configs/config.yaml not found")
except Exception as e:
    check("config.yaml valid", False, str(e))

# ── 5. Dataset Directories ─────────────────────────────────────────────────
print("\n[ Dataset Directories ]")
try:
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    P = cfg["paths"]

    dirs_to_check = [
        (P["iq_oth_nccd"],       "IQ-OTH/NCCD root"),
        (P["ct_lung_finder"],    "CT Lung Finder root"),
        (P["nodule_malignancy"], "Nodule Malignancy root"),
    ]
    for path, label in dirs_to_check:
        exists = os.path.isdir(path)
        check(label, exists,
              path if exists else f"{path}  (run: python download_data.py)",
              warn_only=not exists)

    # Check class sub-folders for classifier
    for cls in ["Normal", "Benign", "Malignant"]:
        cls_path = os.path.join(P["iq_oth_nccd"], cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path)
                     if f.endswith((".png",".jpg",".jpeg"))])
            check(f"  {cls}/", True, f"{n} images")
        else:
            check(f"  {cls}/", False,
                  "folder missing — check dataset structure", warn_only=True)

except Exception as e:
    check("Dataset paths", False, str(e))

# ── 6. Dummy Model Forward Pass ────────────────────────────────────────────
print("\n[ Model Architecture ]")
try:
    import torch
    import timm

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=3)
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    check("EfficientNet-B4 forward pass",
          out.shape == (1, 3),
          f"Output shape: {tuple(out.shape)}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    check("EfficientNet-B4 parameters", True, f"{total_params:.1f}M parameters")

except Exception as e:
    check("EfficientNet-B4 forward pass", False, str(e))

try:
    from ultralytics import YOLO
    check("YOLOv8 importable", True, "Ultralytics OK")
except Exception as e:
    check("YOLOv8 importable", False, str(e))

# ── 7. Streamlit ───────────────────────────────────────────────────────────
print("\n[ Web App ]")
try:
    import streamlit as st
    check("Streamlit importable", True, f"v{st.__version__}")
    check("App file exists",
          os.path.isfile("app/app.py"),
          "app/app.py")
    check("Inference file exists",
          os.path.isfile("app/inference.py"),
          "app/inference.py")
except Exception as e:
    check("Streamlit importable", False, str(e))

# ── 8. Trained Weights ─────────────────────────────────────────────────────
print("\n[ Trained Model Weights ]")
try:
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    P = cfg["paths"]

    clf_path = os.path.join(P["classifier_output"], "best_model.pth")
    det_path = os.path.join(P["detector_output"], "runs", "weights", "best.pt")

    check("Classifier weights",
          os.path.isfile(clf_path), clf_path,
          warn_only=not os.path.isfile(clf_path))
    check("Detector weights",
          os.path.isfile(det_path), det_path,
          warn_only=not os.path.isfile(det_path))

    if not os.path.isfile(clf_path):
        print(f"    → Run: python models/classifier/train_classifier.py")
    if not os.path.isfile(det_path):
        print(f"    → Run: python models/detector/train_detector.py")

except Exception as e:
    check("Trained weights check", False, str(e))

# ── Final Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(results)
total  = len(results)
print(f"  Result: {passed}/{total} checks passed")

if passed == total:
    print(f"\n  {PASS}  ALL CHECKS PASSED — You are ready to train!")
    print("\n  Recommended order:")
    print("    1. python download_data.py")
    print("    2. python notebooks/01_eda.py")
    print("    3. python models/detector/convert_to_yolo.py")
    print("    4. python models/classifier/train_classifier.py")
    print("    5. python models/detector/train_detector.py")
    print("    6. python notebooks/02_evaluate.py")
    print("    7. streamlit run app/app.py")
else:
    failed = total - passed
    print(f"\n  {FAIL}  {failed} check(s) failed — fix them before training.")
    print("  Look for ✗ lines above and follow the instructions.")

print("=" * 60)


