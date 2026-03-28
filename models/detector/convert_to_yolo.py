"""
convert_to_yolo.py  —  chest-ctscan-images → YOLO format conversion
─────────────────────────────────────────────────────────────────────────────
Dataset: mohamedhanyyy/chest-ctscan-images

Actual folder structure:
    train/
        adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/   ← long clinical name
        large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/← long clinical name
        squamous.cell.carcinoma/
        normal/
    valid/                                              ← remapped to "val"
        adenocarcinoma/
        large.cell.carcinoma/
        squamous.cell.carcinoma/
        normal/
    test/
        adenocarcinoma/
        large.cell.carcinoma/
        squamous.cell.carcinoma/
        normal/

Class mapping (prefix-based to handle long train folder names):
    "adenocarcinoma..."          → class 0
    "large.cell.carcinoma..."    → class 1
    "squamous.cell.carcinoma..." → class 2
    "normal"                     → empty label (background)
─────────────────────────────────────────────────────────────────────────────
"""

import shutil
import yaml
from pathlib import Path
from tqdm import tqdm


# ── Class prefix mapping ───────────────────────────────────────────────────
# We use startswith() because the train/ folders have long clinical suffixes.
# Order matters — more specific prefixes must come before broader ones.
CLASS_PREFIXES = [
    ("large.cell.carcinoma", 1),       # must come before any future "large" match
    ("squamous.cell.carcinoma",  2),
    ("adenocarcinoma",           0),   # broad prefix — keep last among cancer classes
]

# The dataset's split folder names → YOLO split names
SPLIT_MAP = {
    "train": "train",
    "valid": "val",    # Kaggle uses "valid", Ultralytics expects "val"
    "test":  "test",
}


def resolve_class(folder_name: str):
    """
    Map a folder name to a YOLO class index using prefix matching.
    Returns None  for 'normal' (background — empty label file).
    Returns -1    for completely unknown folders (will be skipped).
    Returns 0/1/2 for the three cancer classes.
    """
    name_lower = folder_name.lower()

    if name_lower.startswith("normal"):
        return None  # background

    for prefix, class_id in CLASS_PREFIXES:
        if name_lower.startswith(prefix):
            return class_id

    return -1  # unknown / skip


def build_yolo_dataset(dataset_root: str, output_dir: str):
    """
    Reorganise mohamedhanyyy/chest-ctscan-images into the YOLO directory
    structure that Ultralytics expects:

        yolo_dataset/
            images/  train/  val/  test/
            labels/  train/  val/  test/
            dataset.yaml
    """
    dataset_root = Path(dataset_root)
    output_path  = Path(output_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            f"Download from Kaggle: mohamedhanyyy/chest-ctscan-images\n"
            f"Then extract so the path above contains train/, valid/, test/ folders."
        )

    # Create output directories
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_cancer = 0
    total_normal = 0
    split_counts = {}

    for src_split, dst_split in SPLIT_MAP.items():
        split_dir = dataset_root / src_split
        if not split_dir.exists():
            print(f"  Warning: split folder not found → {split_dir}, skipping.")
            continue

        images_in_split = 0
        print(f"\nProcessing split: '{src_split}' → '{dst_split}'")

        # List all subfolders and print them so the user can verify mapping
        subfolders = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"  Found {len(subfolders)} class folders:")
        for sf in subfolders:
            cls = resolve_class(sf.name)
            if cls is None:
                label = "NORMAL (empty label)"
            elif cls == -1:
                label = "UNKNOWN (will skip)"
            else:
                label = f"class {cls}"
            print(f"    {sf.name}  →  {label}")

        for class_folder in subfolders:
            class_id = resolve_class(class_folder.name)

            if class_id == -1:
                print(f"  Skipping unknown folder: {class_folder.name}")
                continue

            is_normal = (class_id is None)

            image_files = [
                p for p in class_folder.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]

            for img_path in tqdm(image_files, desc=f"    {class_folder.name[:40]}"):
                stem = img_path.stem
                ext  = img_path.suffix.lower()

                # Prefix stem with class folder name to avoid filename collisions
                # (train and test/val can have images with the same base filename)
                safe_stem = f"{class_folder.name}_{stem}"

                # Copy image
                dest_img = output_path / "images" / dst_split / f"{safe_stem}{ext}"
                shutil.copy2(img_path, dest_img)

                # Write label file
                dest_lbl = output_path / "labels" / dst_split / f"{safe_stem}.txt"
                if is_normal:
                    dest_lbl.write_text("")            # empty = background
                    total_normal += 1
                else:
                    # Full-image proxy bounding box (no real bbox annotations exist)
                    dest_lbl.write_text(
                        f"{class_id} 0.500000 0.500000 1.000000 1.000000\n"
                    )
                    total_cancer += 1

                images_in_split += 1
                total_images    += 1

        split_counts[dst_split] = images_in_split

    # ── Write dataset.yaml ─────────────────────────────────────────────────
    dataset_yaml_content = {
        "path":  str(output_path.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    3,
        "names": ["adenocarcinoma", "large_cell_carcinoma", "squamous_cell_carcinoma"],
    }

    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml_content, f, default_flow_style=False, sort_keys=False)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Conversion complete!")
    print(f"  Total images processed  : {total_images}")
    print(f"  Cancer images (with box): {total_cancer}")
    print(f"  Normal images (no box)  : {total_normal}")
    print(f"  Split breakdown         : {split_counts}")
    print(f"  YOLO dataset saved to   : {output_path}")
    print(f"  Dataset YAML            : {yaml_path}")

    return str(yaml_path)


if __name__ == "__main__":
    import sys
    import yaml as pyyaml

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    with open("configs/config.yaml") as f:
        cfg = pyyaml.safe_load(f)

    build_yolo_dataset(
        dataset_root = cfg["paths"]["chest_ctscan"],
        output_dir   = cfg["paths"]["yolo_dataset"],
    )


