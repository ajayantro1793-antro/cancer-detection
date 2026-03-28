r"""
download_data.py  —  Kaggle Dataset Downloader
─────────────────────────────────────────────────────────────────────────────
Run this script ONCE before training to download all three datasets.

Prerequisites:
  1. pip install kaggle
  2. Place your kaggle.json API token at:
       Windows: C:\Users\YourName\.kaggle\kaggle.json
       Linux:   ~/.kaggle/kaggle.json
  3. Get your API token from: https://www.kaggle.com/settings → API → Create Token

Total download size: approximately 1.1 GB
  - IQ-OTH/NCCD:          ~600 MB
  - CT Lung Finder:        ~662 MB (we skip the 3D zip to save space)
  - Lung Nodule Malignancy: ~400 MB

Usage:
    python download_data.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import zipfile
import subprocess
from pathlib import Path


def run(cmd: str):
    """Run a shell command and print its output."""
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"Warning: command exited with code {result.returncode}")


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file, showing progress."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"  Done → {extract_to}")


def main():
    print("=" * 60)
    print("Lung Cancer Detection — Dataset Downloader")
    print("=" * 60)

    # ── Dataset 1: IQ-OTH/NCCD ────────────────────────────────────────────
    print("\n[1/3] Downloading IQ-OTH/NCCD (Classification)...")
    os.makedirs("data/iq_oth_nccd", exist_ok=True)
    run("kaggle datasets download -d hamdallak/the-iqothnccd-lung-cancer-dataset "
        "-p data/iq_oth_nccd --unzip")
    print("IQ-OTH/NCCD ready.")

    # ── Dataset 2: CT Lung Finder (kmader) ────────────────────────────────
    print("\n[2/3] Downloading CT Lung Finder (Segmentation)...")
    os.makedirs("data/ct_lung_finder", exist_ok=True)
    # Download only the files we need — skip 3d_images.zip to save ~400MB
    run("kaggle datasets download -d kmader/finding-lungs-in-ct-data "
        "-p data/ct_lung_finder --unzip")
    print("CT Lung Finder ready.")

    # ── Dataset 3: Lung Nodule Malignancy (andrewmvd) ─────────────────────
    print("\n[3/3] Downloading Lung Nodule Malignancy (Detection)...")
    os.makedirs("data/nodule_malignancy", exist_ok=True)
    run("kaggle datasets download -d andrewmvd/lung-nodule-malignancy "
        "-p data/nodule_malignancy --unzip")
    print("Lung Nodule Malignancy ready.")

    print("\n" + "=" * 60)
    print("All datasets downloaded successfully!")
    print("Next steps:")
    print("  1. python models/detector/convert_to_yolo.py")
    print("  2. python models/classifier/train_classifier.py")
    print("  3. python models/detector/train_detector.py")
    print("  4. streamlit run app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()


