"""
prepare_data.py — CropGuard AI · Dataset Preparation
======================================================
Run this ONCE after downloading the PlantVillage dataset from Kaggle.

It will:
1. Find the 10 classes your model uses inside the downloaded dataset
2. Split images 85% train / 15% val
3. Copy them into data/train/ and data/val/

Run from D:\\crop_disease_app\\:
    python prepare_data.py
"""

import os
import shutil
import random
from pathlib import Path

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

# Where Kaggle extracted the dataset
# After extraction it's usually one of these — script will auto-detect
POSSIBLE_SOURCE_DIRS = [
    r"data\PlantVillage\PlantVillage",   # nested — your exact layout
    r"data\PlantVillage",
    r"data\plant-disease",
    r"data\plantdisease",
    r"data",
]

# Exact 10 classes your model was trained on
# These must match class_names.json exactly
TARGET_CLASSES = [
    "Pepper___Bacterial_Spot",
    "Pepper___Healthy",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Tomato___Bacterial_Spot",
    "Tomato___Early_Blight",
    "Tomato___Healthy",
    "Tomato___Late_Blight",
    "Tomato___Yellow_Leaf_Curl_Virus",
]

# The Kaggle dataset uses slightly different folder names (spaces, underscores)
# This mapping handles all known variants
CLASS_NAME_MAP = {
    # Pepper — your dataset uses double underscore: Pepper__bell___...
    "Pepper___Bacterial_Spot":      ["Pepper__bell___Bacterial_spot",
                                     "Pepper__bell___Bacterial_Spot",
                                     "Pepper___Bacterial_spot",
                                     "Pepper___Bacterial_Spot"],
    "Pepper___Healthy":             ["Pepper__bell___healthy",
                                     "Pepper__bell___Healthy",
                                     "Pepper___healthy",
                                     "Pepper___Healthy"],
    # Potato — your dataset uses Early_blight / healthy / Late_blight
    "Potato___Early_Blight":        ["Potato___Early_blight", "Potato___Early_Blight"],
    "Potato___Healthy":             ["Potato___healthy", "Potato___Healthy"],
    "Potato___Late_Blight":         ["Potato___Late_blight", "Potato___Late_Blight"],
    # Tomato — your dataset uses single underscore
    "Tomato___Bacterial_Spot":      ["Tomato_Bacterial_spot", "Tomato_Bacterial_Spot",
                                     "Tomato___Bacterial_spot", "Tomato___Bacterial_Spot"],
    "Tomato___Early_Blight":        ["Tomato_Early_blight", "Tomato_Early_Blight",
                                     "Tomato___Early_blight", "Tomato___Early_Blight"],
    "Tomato___Healthy":             ["Tomato_healthy", "Tomato_Healthy",
                                     "Tomato___healthy", "Tomato___Healthy"],
    "Tomato___Late_Blight":         ["Tomato_Late_blight", "Tomato_Late_Blight",
                                     "Tomato___Late_blight", "Tomato___Late_Blight"],
    "Tomato___Yellow_Leaf_Curl_Virus": ["Tomato__Tomato_YellowLeaf__Curl_Virus",
                                        "Tomato_YellowLeaf_Curl_Virus",
                                        "Tomato___Yellow_Leaf_Curl_Virus",
                                        "Tomato__YellowLeaf__Curl_Virus"],
}

TRAIN_SPLIT = 0.85   # 85% train, 15% val
SEED        = 42
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

OUTPUT_TRAIN = Path("data/train")
OUTPUT_VAL   = Path("data/val")


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def find_source_dir():
    """Find the PlantVillage folder that contains class subfolders."""
    for d in POSSIBLE_SOURCE_DIRS:
        p = Path(d)
        if p.is_dir():
            # Check it has at least one of our expected class folders (any variant)
            subdirs = [x.name for x in p.iterdir() if x.is_dir()]
            for variants in CLASS_NAME_MAP.values():
                for v in variants:
                    if v in subdirs:
                        return p
    return None


def find_class_folder(source_dir: Path, class_name: str):
    """
    Find the actual folder in the dataset for a given class.
    Handles case differences and name variants.
    """
    subdirs = {x.name: x for x in source_dir.iterdir() if x.is_dir()}

    # Try exact match first
    if class_name in subdirs:
        return subdirs[class_name]

    # Try known variants
    for variant in CLASS_NAME_MAP.get(class_name, []):
        if variant in subdirs:
            return subdirs[variant]

    # Try case-insensitive match
    lower_map = {k.lower(): v for k, v in subdirs.items()}
    if class_name.lower() in lower_map:
        return lower_map[class_name.lower()]

    return None


def get_images(folder: Path):
    return [f for f in folder.iterdir()
            if f.is_file() and f.suffix in IMG_EXTS]


def copy_split(images, train_dir: Path, val_dir: Path, class_name: str):
    random.shuffle(images)
    n_train = int(len(images) * TRAIN_SPLIT)

    train_class_dir = train_dir / class_name
    val_class_dir   = val_dir   / class_name
    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    for img in images[:n_train]:
        shutil.copy2(img, train_class_dir / img.name)
    for img in images[n_train:]:
        shutil.copy2(img, val_class_dir / img.name)

    return n_train, len(images) - n_train


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    print("=" * 60)
    print("CropGuard AI — Dataset Preparation")
    print("=" * 60)

    # Step 1: Find source
    source_dir = find_source_dir()
    if source_dir is None:
        print("\nERROR: Could not find the PlantVillage dataset folder.")
        print("Make sure you extracted plantdisease.zip into the data/ folder.")
        print("Expected one of:", POSSIBLE_SOURCE_DIRS)
        print("\nYour data/ folder currently contains:")
        for p in Path("data").iterdir():
            print("  ", p)
        return

    print(f"\nFound dataset at: {source_dir}")
    print(f"Output train dir: {OUTPUT_TRAIN}")
    print(f"Output val dir:   {OUTPUT_VAL}")
    print(f"Train/val split:  {int(TRAIN_SPLIT*100)}% / {int((1-TRAIN_SPLIT)*100)}%")
    print()

    total_train = 0
    total_val   = 0
    missing     = []

    for class_name in TARGET_CLASSES:
        folder = find_class_folder(source_dir, class_name)

        if folder is None:
            print(f"  MISSING: {class_name}  <-- not found in dataset")
            missing.append(class_name)
            continue

        images = get_images(folder)
        if not images:
            print(f"  EMPTY:   {class_name}  (folder found but no images)")
            missing.append(class_name)
            continue

        n_train, n_val = copy_split(images, OUTPUT_TRAIN, OUTPUT_VAL, class_name)
        total_train   += n_train
        total_val     += n_val
        print(f"  OK  {class_name:<40} {len(images):>5} images  "
              f"({n_train} train / {n_val} val)")

    print()
    print("=" * 60)
    print(f"Total images copied:  {total_train + total_val}")
    print(f"  Training set:       {total_train}")
    print(f"  Validation set:     {total_val}")

    if missing:
        print(f"\nWARNING: {len(missing)} class(es) not found:")
        for m in missing:
            print(f"  - {m}")
        print("The retrain script will fail for missing classes.")
        print("Check the actual folder names inside your dataset and update CLASS_NAME_MAP.")
    else:
        print("\nAll 10 classes found and organized successfully.")
        print("\nNext step — run the retraining script:")
        print("  python retrain.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
