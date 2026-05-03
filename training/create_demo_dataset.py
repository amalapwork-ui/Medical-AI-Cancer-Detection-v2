# %% ============================================================
# create_demo_dataset.py
# Run this entire file in Google Colab to:
#   1. Install dependencies + configure Kaggle
#   2. Download all three datasets
#   3. Analyse folder structures
#   4. Build demo_dataset/ (20 diverse images per class, 224×224 RGB)
#   5. Validate + visualise
#   6. Zip and download
# ============================================================

# %% CELL 1 — Install / upgrade dependencies
# ──────────────────────────────────────────
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "kaggle", "pillow", "numpy", "matplotlib"], check=True)

print("Dependencies ready.")


# %% CELL 2 — Kaggle credentials
# ──────────────────────────────
# Upload your kaggle.json via the Files panel, or use the uploader below.

try:
    from google.colab import files as colab_files
    print("Upload your kaggle.json now:")
    colab_files.upload()
except ImportError:
    print("Not in Colab — assuming kaggle.json is already in ~/.kaggle/")

import os, shutil
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# Move kaggle.json if it was uploaded to the working directory
if os.path.exists("kaggle.json"):
    shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))

os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
print("Kaggle credentials configured.")


# %% CELL 3 — Download datasets
# ──────────────────────────────
from pathlib import Path

DATA_ROOT = Path("/content/datasets")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

datasets = [
    ("masoudnickparvar/brain-tumor-mri-dataset",      DATA_ROOT / "brain"),
    ("hamdallak/the-iqothnccd-lung-cancer-dataset",   DATA_ROOT / "lung"),
    ("kmader/colorectal-histology-mnist",              DATA_ROOT / "colon"),
]

for slug, dest in datasets:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {slug} → {dest} …")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug,
         "-p", str(dest), "--unzip"],
        check=True,
    )
    print(f"  Done.")

print("\nAll downloads complete.")


# %% CELL 4 — Analyse folder structures
# ──────────────────────────────────────
import os

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def count_images(folder: Path) -> int:
    return sum(
        1 for f in folder.rglob("*")
        if f.suffix.lower() in IMAGE_EXTS
    )

def print_tree(root: Path, max_depth: int = 3, _depth: int = 0) -> None:
    if _depth > max_depth:
        return
    indent = "  " * _depth
    items = sorted(root.iterdir()) if root.is_dir() else []
    for item in items:
        if item.is_dir():
            n = count_images(item)
            print(f"{indent}📁 {item.name}/   ({n} images)")
            print_tree(item, max_depth, _depth + 1)
        else:
            if _depth <= 2 and item.suffix.lower() in IMAGE_EXTS:
                print(f"{indent}🖼  {item.name}")

print("=" * 60)
for name, root in [("BRAIN", DATA_ROOT/"brain"),
                   ("LUNG",  DATA_ROOT/"lung"),
                   ("COLON", DATA_ROOT/"colon")]:
    print(f"\n{'='*60}")
    print(f"  {name} dataset  ({root})")
    print("=" * 60)
    print_tree(root, max_depth=3)


# %% CELL 5 — Class-folder discovery helpers
# ──────────────────────────────────────────

# ── Brain ─────────────────────────────────────────────────────
BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def find_brain_folders(root: Path) -> dict[str, Path]:
    """
    The brain dataset has Training/ and Testing/ splits.
    We pool both to maximise variety in the demo set.
    """
    found: dict[str, list[Path]] = {c: [] for c in BRAIN_CLASSES}
    for split_dir in root.rglob("*"):
        if not split_dir.is_dir():
            continue
        name = split_dir.name.lower()
        if name in found:
            found[name].append(split_dir)
    # flatten: collect all image paths per class
    result: dict[str, list[Path]] = {}
    for cls, dirs in found.items():
        imgs = []
        for d in dirs:
            imgs += [f for f in sorted(d.iterdir())
                     if f.suffix.lower() in IMAGE_EXTS]
        if imgs:
            result[cls] = imgs
    return result


# ── Lung ──────────────────────────────────────────────────────
LUNG_FOLDER_MAP = {
    "benign"       : "benign",
    "benign cases" : "benign",
    "benign case"  : "benign",
    "bengin cases" : "benign",   # typo variant in dataset
    "bengin case"  : "benign",
    "malignant"       : "malignant",
    "malignant cases" : "malignant",
    "malignant case"  : "malignant",
    "normal"       : "normal",
    "normal cases" : "normal",
    "normal case"  : "normal",
}
LUNG_CLASSES = ["benign", "malignant", "normal"]

def find_lung_folders(root: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {c: [] for c in LUNG_CLASSES}
    # Walk every directory; match name after lowercasing + stripping
    for p in sorted(root.rglob("*")):
        if not p.is_dir():
            continue
        key = p.name.lower().strip()
        if key in LUNG_FOLDER_MAP:
            label = LUNG_FOLDER_MAP[key]
            imgs = [f for f in sorted(p.iterdir())
                    if f.suffix.lower() in IMAGE_EXTS]
            found[label].extend(imgs)
    # Deduplicate (same file might be referenced twice)
    return {k: list(dict.fromkeys(v)) for k, v in found.items() if v}


# ── Colon (Kather 2016) ───────────────────────────────────────
COLON_CLASSES = ["adipose", "complex", "debris", "empty",
                 "lympho", "mucosa", "stroma", "tumor"]

COLON_FOLDER_MAP = {
    # Numbered variants (Kather 2016 official)
    "01_tumor"   : "tumor",   "tumor"   : "tumor",   "tum"     : "tumor",
    "02_stroma"  : "stroma",  "stroma"  : "stroma",  "str"     : "stroma",
    "03_complex" : "complex", "complex" : "complex",
    "04_lympho"  : "lympho",  "lympho"  : "lympho",  "lym"     : "lympho",
    "05_debris"  : "debris",  "debris"  : "debris",  "deb"     : "debris",
    "06_mucosa"  : "mucosa",  "mucosa"  : "mucosa",  "norm"    : "mucosa",
    "07_adipose" : "adipose", "adipose" : "adipose", "adi"     : "adipose",
    "08_empty"   : "empty",   "empty"   : "empty",   "back"    : "empty",
    # Alternative capitalisations
    "tumor"      : "tumor",   "stroma"  : "stroma",
    "lymphocyte" : "lympho",  "mucosa"  : "mucosa",
}

def find_colon_folders(root: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {c: [] for c in COLON_CLASSES}
    for p in sorted(root.rglob("*")):
        if not p.is_dir():
            continue
        key = p.name.lower().strip()
        if key in COLON_FOLDER_MAP:
            label = COLON_FOLDER_MAP[key]
            imgs = [f for f in sorted(p.iterdir())
                    if f.suffix.lower() in IMAGE_EXTS]
            found[label].extend(imgs)
    return {k: list(dict.fromkeys(v)) for k, v in found.items() if v}


# Verify discovery
print("Brain class folders found:")
brain_imgs = find_brain_folders(DATA_ROOT / "brain")
for cls in BRAIN_CLASSES:
    n = len(brain_imgs.get(cls, []))
    status = "✓" if n > 0 else "✗ MISSING"
    print(f"  {cls:15s}: {n:4d} images  {status}")

print("\nLung class folders found:")
lung_imgs = find_lung_folders(DATA_ROOT / "lung")
for cls in LUNG_CLASSES:
    n = len(lung_imgs.get(cls, []))
    status = "✓" if n > 0 else "✗ MISSING"
    print(f"  {cls:15s}: {n:4d} images  {status}")

print("\nColon class folders found:")
colon_imgs = find_colon_folders(DATA_ROOT / "colon")
for cls in COLON_CLASSES:
    n = len(colon_imgs.get(cls, []))
    status = "✓" if n > 0 else "✗ MISSING"
    print(f"  {cls:15s}: {n:4d} images  {status}")


# %% CELL 6 — Build demo_dataset/
# ────────────────────────────────
import numpy as np
from PIL import Image, UnidentifiedImageError

DEMO_ROOT   = Path("/content/demo_dataset")
IMAGES_PER_CLASS = 20
OUTPUT_SIZE = (224, 224)

def select_diverse(image_paths: list[Path], n: int) -> list[Path]:
    """
    Pick n evenly-spaced images from a sorted list.
    Ensures spatial diversity: no sequential duplicates, full range covered.
    """
    if len(image_paths) <= n:
        return image_paths
    indices = np.linspace(0, len(image_paths) - 1, n, dtype=int)
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for i in indices:
        if i not in seen:
            result.append(image_paths[i])
            seen.add(i)
    return result

def save_image(src: Path, dst: Path) -> bool:
    """
    Open src → convert to RGB → resize to OUTPUT_SIZE → save as JPEG.
    Returns True on success, False on error (corrupted / unreadable file).
    """
    try:
        img = Image.open(src)
        img.load()                         # full decode — catches truncated files
        img = img.convert("RGB")
        img = img.resize(OUTPUT_SIZE, Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, format="JPEG", quality=92)
        return True
    except (UnidentifiedImageError, OSError, Exception):
        return False

def build_cancer_split(
    cancer_name: str,
    class_image_map: dict[str, list[Path]],
    classes: list[str],
    n: int = IMAGES_PER_CLASS,
) -> dict[str, int]:
    """
    Copy n diverse images per class into demo_dataset/<cancer_name>/<class>/.
    Returns a dict of {class: images_saved}.
    """
    split_root = DEMO_ROOT / cancer_name
    counts: dict[str, int] = {}

    for cls in classes:
        imgs = class_image_map.get(cls, [])
        if not imgs:
            print(f"  [WARNING] {cancer_name}/{cls}: no images found — skipping")
            counts[cls] = 0
            continue

        selected = select_diverse(sorted(imgs), n)
        dst_dir  = split_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, src in enumerate(selected):
            dst = dst_dir / f"{cls}_{i+1:02d}.jpg"
            if save_image(src, dst):
                saved += 1
            else:
                print(f"    [SKIP] Corrupt or unreadable: {src.name}")

        counts[cls] = saved
        print(f"  {cls:20s}: saved {saved}/{len(selected)}")

    return counts

# Remove any previous run
if DEMO_ROOT.exists():
    shutil.rmtree(DEMO_ROOT)

print(f"\nBuilding demo_dataset/ → {DEMO_ROOT}\n")

print("─── Brain ────────────────────────────")
brain_counts = build_cancer_split("brain", brain_imgs, BRAIN_CLASSES)

print("\n─── Lung ─────────────────────────────")
lung_counts  = build_cancer_split("lung",  lung_imgs,  LUNG_CLASSES)

print("\n─── Colon ────────────────────────────")
colon_counts = build_cancer_split("colon", colon_imgs, COLON_CLASSES)

print("\nDone.")


# %% CELL 7 — Validate: counts, paths, sample visualisation
# ──────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def validate_split(cancer_name: str, classes: list[str]) -> int:
    split_root = DEMO_ROOT / cancer_name
    total = 0
    print(f"\n{'─'*50}")
    print(f"  {cancer_name.upper()}")
    print(f"{'─'*50}")
    for cls in classes:
        cls_dir = split_root / cls
        if not cls_dir.exists():
            print(f"  {cls:20s}: MISSING DIRECTORY")
            continue
        imgs = sorted(cls_dir.glob("*.jpg"))
        n = len(imgs)
        total += n
        ok = "✓" if n == IMAGES_PER_CLASS else f"⚠ expected {IMAGES_PER_CLASS}"
        print(f"  {cls:20s}: {n:3d} images  {ok}")
        if imgs:
            print(f"    first: {imgs[0].name}   last: {imgs[-1].name}")
    print(f"  {'SUBTOTAL':20s}: {total}")
    return total

grand_total = 0
grand_total += validate_split("brain", BRAIN_CLASSES)
grand_total += validate_split("lung",  LUNG_CLASSES)
grand_total += validate_split("colon", COLON_CLASSES)

print(f"\n{'='*50}")
print(f"  GRAND TOTAL: {grand_total} images")
print(f"  Expected   : {(len(BRAIN_CLASSES) + len(LUNG_CLASSES) + len(COLON_CLASSES)) * IMAGES_PER_CLASS}")
print(f"{'='*50}")


# ── Visualise sample images ────────────────────────────────
def show_samples(cancer_name: str, classes: list[str], cols: int = 4) -> None:
    rows  = len(classes)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f"{cancer_name.upper()} — sample images (224×224)", fontsize=14)

    for r, cls in enumerate(classes):
        cls_dir = DEMO_ROOT / cancer_name / cls
        imgs    = sorted(cls_dir.glob("*.jpg")) if cls_dir.exists() else []
        for c in range(cols):
            ax = axes[r][c] if rows > 1 else axes[c]
            if c < len(imgs):
                ax.imshow(mpimg.imread(str(imgs[c])))
                if c == 0:
                    ax.set_ylabel(cls, fontsize=10, rotation=0,
                                  labelpad=60, va="center")
            else:
                ax.axis("off")
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"/content/samples_{cancer_name}.png", dpi=80, bbox_inches="tight")
    plt.show()
    print(f"  Saved /content/samples_{cancer_name}.png")

print("\nVisualising brain samples …")
show_samples("brain", BRAIN_CLASSES, cols=5)

print("\nVisualising lung samples …")
show_samples("lung",  LUNG_CLASSES,  cols=5)

print("\nVisualising colon samples …")
show_samples("colon", COLON_CLASSES, cols=5)


# %% CELL 8 — Verify all images are valid (PIL can re-open them)
# ──────────────────────────────────────────────────────────────
print("\nRunning integrity check on all saved images …")
bad_files = []
total_checked = 0

for img_path in sorted(DEMO_ROOT.rglob("*.jpg")):
    total_checked += 1
    try:
        with Image.open(img_path) as im:
            im.load()
        w, h = im.size
        if w != 224 or h != 224:
            bad_files.append((img_path, f"wrong size {w}×{h}"))
        elif im.mode != "RGB":
            bad_files.append((img_path, f"wrong mode {im.mode}"))
    except Exception as e:
        bad_files.append((img_path, str(e)))

print(f"Checked {total_checked} images.")
if bad_files:
    print(f"PROBLEMS ({len(bad_files)}):")
    for path, reason in bad_files:
        print(f"  {path.relative_to(DEMO_ROOT)}  →  {reason}")
else:
    print("All images OK  (224×224 RGB JPEG) ✓")


# %% CELL 9 — Print final folder structure
# ─────────────────────────────────────────
print("\nFinal demo_dataset/ structure:")
print("=" * 50)
for cancer in ["brain", "lung", "colon"]:
    cancer_dir = DEMO_ROOT / cancer
    if not cancer_dir.exists():
        continue
    print(f"\n  {cancer}/")
    for cls_dir in sorted(cancer_dir.iterdir()):
        imgs = list(cls_dir.glob("*.jpg"))
        print(f"    {cls_dir.name}/  ({len(imgs)} images)")
        # Show first 3 filenames as examples
        for img in imgs[:3]:
            print(f"      {img.name}")
        if len(imgs) > 3:
            print(f"      … ({len(imgs) - 3} more)")


# %% CELL 10 — Create ZIP archive and download
# ──────────────────────────────────────────────
ZIP_PATH = "/content/demo_dataset"

print(f"\nZipping {DEMO_ROOT} …")
shutil.make_archive(ZIP_PATH, "zip", str(DEMO_ROOT.parent), DEMO_ROOT.name)

zip_file = Path(ZIP_PATH + ".zip")
size_mb  = zip_file.stat().st_size / (1024 * 1024)
print(f"Created {zip_file}  ({size_mb:.1f} MB)")

try:
    from google.colab import files as colab_files
    print("Downloading demo_dataset.zip …")
    colab_files.download(str(zip_file))
except ImportError:
    print(f"(Not in Colab) ZIP is at: {zip_file}")

print("\nAll done. demo_dataset.zip is ready.")
