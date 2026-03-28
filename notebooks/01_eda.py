"""
01_eda.py  —  Exploratory Data Analysis
─────────────────────────────────────────────────────────────────────────────
Run this BEFORE training to understand your datasets:
  - Class distribution (are classes balanced?)
  - Sample images from each class
  - Pixel intensity distributions (important for CT windowing choices)
  - Nodule size distribution from the detection dataset
  - Mask coverage from the CT Lung Finder dataset

Run with:
    python notebooks/01_eda.py

Outputs saved to: notebooks/eda_outputs/
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

P = cfg["paths"]
OUT = Path("notebooks/eda_outputs")
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = ["Normal", "Benign", "Malignant"]
CLASS_COLORS = {"Normal": "#059669", "Benign": "#D97706", "Malignant": "#EF4444"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
})


# ── 1. Class Distribution ──────────────────────────────────────────────────

def plot_class_distribution():
    print("\n[1/5] Analysing class distribution...")
    counts = {}
    for cls in CLASS_NAMES:
        cls_dir = Path(P["iq_oth_nccd"]) / cls
        if cls_dir.exists():
            files = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
            counts[cls] = len(files)
        else:
            counts[cls] = 0
            print(f"  Warning: {cls_dir} not found")

    total = sum(counts.values())
    print(f"  Total images: {total}")
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt} ({100*cnt/total:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IQ-OTH/NCCD — Class Distribution", fontsize=14, fontweight="bold")

    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    bars = ax1.bar(CLASS_NAMES,
                   [counts[c] for c in CLASS_NAMES],
                   color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_title("Image Count per Class")
    ax1.set_ylabel("Number of Images")
    for bar, (cls, cnt) in zip(bars, counts.items()):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 2,
                 f"{cnt}\n({100*cnt/total:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    wedges, texts, autotexts = ax2.pie(
        [counts[c] for c in CLASS_NAMES],
        labels=CLASS_NAMES, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    ax2.set_title("Class Balance")

    plt.tight_layout()
    plt.savefig(OUT / "01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 01_class_distribution.png")
    return counts


# ── 2. Sample Images from Each Class ─────────────────────────────────────

def plot_sample_images(n_per_class: int = 4):
    print("\n[2/5] Plotting sample images...")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Sample CT Slices — IQ-OTH/NCCD Dataset",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(len(CLASS_NAMES), n_per_class, figure=fig,
                           hspace=0.4, wspace=0.1)

    for row_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = Path(P["iq_oth_nccd"]) / cls
        if not cls_dir.exists():
            continue

        images = sorted(list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")))
        sample = images[:n_per_class]

        for col_idx, img_path in enumerate(sample):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(f"{cls}",
                             color=CLASS_COLORS[cls],
                             fontsize=13, fontweight="bold", loc="left")

    plt.savefig(OUT / "02_sample_images.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 02_sample_images.png")


# ── 3. Pixel Intensity Distributions ─────────────────────────────────────

def plot_intensity_distributions(n_samples: int = 50):
    print("\n[3/5] Analysing pixel intensity distributions...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Pixel Intensity Distribution by Class\n"
                 "(Understanding why HU windowing matters)",
                 fontsize=13, fontweight="bold")

    for ax, cls in zip(axes, CLASS_NAMES):
        cls_dir = Path(P["iq_oth_nccd"]) / cls
        if not cls_dir.exists():
            ax.set_title(f"{cls} (not found)")
            continue

        images = sorted(list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")))
        sample = images[:n_samples]

        all_pixels = []
        for img_path in sample:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_pixels.extend(img.flatten().tolist())

        all_pixels = np.array(all_pixels)

        ax.hist(all_pixels, bins=80, color=CLASS_COLORS[cls],
                alpha=0.8, edgecolor="none", density=True)
        ax.set_title(f"{cls}\n(n={len(sample)} images)",
                     color=CLASS_COLORS[cls], fontweight="bold")
        ax.set_xlabel("Pixel Intensity (0–255)")
        ax.axvline(all_pixels.mean(), color="black", linestyle="--",
                   linewidth=1.5, label=f"Mean={all_pixels.mean():.0f}")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(OUT / "03_intensity_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 03_intensity_distributions.png")


# ── 4. Nodule Size Distribution (Detection Dataset) ───────────────────────

def plot_nodule_sizes():
    print("\n[4/5] Analysing nodule sizes from detection dataset...")

    annotations_dir = Path(P["nodule_malignancy"]) / "annotations"
    if not annotations_dir.exists():
        print(f"  Skipping — {annotations_dir} not found. Download dataset first.")
        return

    xml_files = list(annotations_dir.glob("*.xml"))
    print(f"  Found {len(xml_files)} annotation files")

    nodule_widths_mm  = []
    nodule_heights_mm = []
    nodules_per_image = []

    for xml_path in tqdm(xml_files, desc="  Parsing XMLs"):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size_elem = root.find("size")
        img_w = int(size_elem.find("width").text)  if size_elem else 512
        img_h = int(size_elem.find("height").text) if size_elem else 512

        objects = root.findall("object")
        nodules_per_image.append(len(objects))

        for obj in objects:
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            # Approx 1 pixel ≈ 0.7mm on standard CT
            nodule_widths_mm.append( (xmax - xmin) * 0.7)
            nodule_heights_mm.append((ymax - ymin) * 0.7)

    nodule_diameters = [(w + h) / 2 for w, h in
                        zip(nodule_widths_mm, nodule_heights_mm)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Lung Nodule Malignancy Dataset — Nodule Statistics",
                 fontsize=13, fontweight="bold")

    # Diameter distribution
    axes[0].hist(nodule_diameters, bins=30, color="#2563EB",
                 alpha=0.8, edgecolor="none")
    axes[0].axvline(6, color="#EF4444", linestyle="--", linewidth=2,
                    label="6mm clinical threshold")
    axes[0].set_title("Nodule Diameter Distribution")
    axes[0].set_xlabel("Estimated Diameter (mm)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    small  = sum(1 for d in nodule_diameters if d < 6)
    medium = sum(1 for d in nodule_diameters if 6 <= d < 20)
    large  = sum(1 for d in nodule_diameters if d >= 20)
    axes[1].bar(["< 6mm\n(micro)", "6–20mm\n(significant)", "≥ 20mm\n(large)"],
                [small, medium, large],
                color=["#059669", "#F59E0B", "#EF4444"],
                edgecolor="white")
    axes[1].set_title("Nodule Size Categories")
    axes[1].set_ylabel("Count")
    for i, v in enumerate([small, medium, large]):
        axes[1].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

    npi_counter = Counter(nodules_per_image)
    axes[2].bar(npi_counter.keys(), npi_counter.values(),
                color="#7C3AED", edgecolor="white")
    axes[2].set_title("Nodules per Image")
    axes[2].set_xlabel("Number of Nodules")
    axes[2].set_ylabel("Number of Images")

    total_nodules = len(nodule_diameters)
    print(f"  Total nodules: {total_nodules}")
    print(f"  Mean diameter: {np.mean(nodule_diameters):.1f} mm")
    print(f"  < 6mm (micro): {small}")
    print(f"  6–20mm:        {medium}")
    print(f"  ≥ 20mm:        {large}")

    plt.tight_layout()
    plt.savefig(OUT / "04_nodule_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 04_nodule_sizes.png")


# ── 5. Lung Mask Coverage Analysis ────────────────────────────────────────

def plot_mask_coverage(n_samples: int = 30):
    print("\n[5/5] Analysing lung mask coverage...")

    masks_dir = Path(P["ct_lung_finder"])
    mask_files = list(masks_dir.rglob("*mask*")) + \
                 list(masks_dir.rglob("*2d_masks*"))

    # Try to find actual mask PNG files
    mask_pngs = []
    for d in masks_dir.rglob("*"):
        if d.is_dir() and "mask" in d.name.lower():
            mask_pngs = list(d.glob("*.png"))[:n_samples]
            break

    if not mask_pngs:
        print(f"  Skipping — mask files not found in {masks_dir}")
        print("  (Download CT Lung Finder dataset first)")
        return

    coverages = []
    for mask_path in tqdm(mask_pngs[:n_samples], desc="  Analysing masks"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            coverage = (mask > 127).sum() / mask.size
            coverages.append(coverage * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(coverages, bins=20, color="#0891B2", alpha=0.8, edgecolor="none")
    ax.set_title("Lung Region Coverage Distribution\n"
                 "(% of image area covered by lung tissue)",
                 fontweight="bold")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Count")
    ax.axvline(np.mean(coverages), color="#EF4444", linestyle="--",
               label=f"Mean = {np.mean(coverages):.1f}%")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT / "05_mask_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Mean lung coverage: {np.mean(coverages):.1f}%")
    print(f"  Saved: 05_mask_coverage.png")


# ── Summary Report ────────────────────────────────────────────────────────

def print_summary(counts: dict):
    print("\n" + "=" * 55)
    print("  EDA SUMMARY")
    print("=" * 55)
    total = sum(counts.values())
    print(f"\n  IQ-OTH/NCCD (Classification)")
    print(f"  {'Class':<12} {'Count':>8} {'Percent':>10}")
    print(f"  {'-'*32}")
    for cls, cnt in counts.items():
        print(f"  {cls:<12} {cnt:>8} {100*cnt/total:>9.1f}%")
    print(f"  {'TOTAL':<12} {total:>8}")

    print(f"\n  All EDA plots saved to: notebooks/eda_outputs/")
    print(f"\n  Recommended next step:")
    print(f"    python models/detector/convert_to_yolo.py")
    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    counts = plot_class_distribution()
    plot_sample_images()
    plot_intensity_distributions()
    plot_nodule_sizes()
    plot_mask_coverage()
    print_summary(counts)


