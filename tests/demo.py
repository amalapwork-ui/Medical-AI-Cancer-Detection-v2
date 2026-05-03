"""
Medical AI Cancer Detection — Demo Image Generator & API Tester
================================================================

Two modes:

  --mode prepare  (run in Google Colab OR locally with Kaggle credentials)
    Downloads all three medical datasets, samples ≥50 images per class,
    generates synthetic non-medical images, packages everything into
    demo_test_images.zip for offline / API testing.

  --mode api      (run locally while FastAPI backend is running)
    Extracts demo_test_images.zip, calls POST /predict for every image,
    generates a full results report and a confusion-matrix summary.

Usage
─────
  # Step 1 — generate zip (in Colab or with kaggle.json present)
  python tests/demo.py --mode prepare

  # Step 2 — test API  (backend must be running: uvicorn backend.app.main:app --reload)
  python tests/demo.py --mode api --api-url http://localhost:8000

  # Custom zip location
  python tests/demo.py --mode api --zip ./demo_test_images.zip

Requirements
────────────
  prepare mode : kaggle, Pillow, numpy
  api mode     : requests, Pillow, numpy, (matplotlib, seaborn for plots)

Google Colab note
─────────────────
  In Colab, run as a script OR paste cells into a notebook.
  Kaggle auth: upload kaggle.json when prompted, or mount Google Drive.
"""

import argparse
import io
import json
import os
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEED          = 42
IMAGES_PER_CLASS = 50    # minimum per medical class
NON_MEDICAL_COUNT = 25   # synthetic non-medical images (validator rejection tests)

ZIP_NAME   = "demo_test_images.zip"
DEMO_DIR   = Path("./demo_test_images")

# Medical class specification
#   cancer_type → list of class folder names (must match config.py CLASS_LABELS)
MEDICAL_CLASSES = {
    "brain": ["glioma", "meningioma", "notumor", "pituitary"],
    "lung" : ["benign", "malignant", "normal"],
    # Kather 2016 — 8 tissue classes (alphabetical)
    "colon": ["adipose", "complex", "debris", "empty", "lympho", "mucosa", "stroma", "tumor"],
}

# Kaggle dataset IDs
KAGGLE_DATASETS = {
    "brain" : "masoudnickparvar/brain-tumor-mri-dataset",
    "lung"  : "hamdallak/the-iqothnccd-lung-cancer-dataset",
    # Kather 2016 — 8-class colorectal histopathology (what the colon model is trained on)
    "colon" : "kmader/colorectal-histology-mnist",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)


def _log(msg: str) -> None:
    print(f"[demo] {msg}", flush=True)


def _ensure_kaggle_auth() -> None:
    """Upload / verify Kaggle credentials."""
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_cfg.exists():
        _log("Kaggle credentials already present.")
        return
    try:
        from google.colab import files as colab_files  # type: ignore
        _log("Running in Colab — please upload kaggle.json:")
        colab_files.upload()
    except ImportError:
        _log("Not in Colab. Place kaggle.json in ~/.kaggle/ and rerun.")
        sys.exit(1)
    os.makedirs(str(Path.home() / ".kaggle"), exist_ok=True)
    os.system("cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
    _log("Kaggle auth complete.")


def _download_dataset(name: str, dst: str) -> None:
    dataset_id = KAGGLE_DATASETS[name]
    _log(f"Downloading {dataset_id} → {dst} …")
    os.system(f"kaggle datasets download -d {dataset_id} -p {dst} --unzip -q")
    _log(f"Download complete: {dst}")


def _collect_images(root: Path, extensions=(".jpg", ".jpeg", ".png", ".bmp")) -> list[Path]:
    imgs = []
    for ext in extensions:
        imgs.extend(root.rglob(f"*{ext}"))
        imgs.extend(root.rglob(f"*{ext.upper()}"))
    return imgs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific samplers
# Each returns a dict: {class_label: [Path, ...]}  with ≥IMAGES_PER_CLASS items
# ─────────────────────────────────────────────────────────────────────────────

def _sample_brain(data_root: Path) -> dict[str, list[Path]]:
    """
    Brain Tumor MRI dataset structure:
      Training/ and Testing/ → each has glioma/, meningioma/, notumor/, pituitary/
    Sample from Testing first (ground-truth labels guaranteed), then Training if short.
    """
    result = {}
    for cls in MEDICAL_CLASSES["brain"]:
        imgs: list[Path] = []
        for split in ["Testing", "Training"]:
            folder = data_root / split / cls
            if folder.exists():
                imgs.extend(_collect_images(folder))
        if not imgs:
            _log(f"  [WARN] brain/{cls}: no images found in {data_root}")
        random.shuffle(imgs)
        result[cls] = imgs[:max(IMAGES_PER_CLASS, len(imgs))]
        _log(f"  brain/{cls:12s}: {len(result[cls])} images collected")
    return result


def _sample_lung(data_root: Path) -> dict[str, list[Path]]:
    """
    IQ-OTH/NCCD structure — folder names may vary.
    Map flexible folder names → clean labels (benign / malignant / normal).
    """
    FOLDER_MAP = {
        "benign": "benign",
        "benign cases": "benign",
        "benign case": "benign",
        "malignant": "malignant",
        "malignant cases": "malignant",
        "malignant case": "malignant",
        "normal": "normal",
    }

    label_to_imgs: dict[str, list[Path]] = {c: [] for c in MEDICAL_CLASSES["lung"]}

    for p in data_root.rglob("*"):
        if not p.is_dir():
            continue
        key = p.name.lower().strip()
        if key in FOLDER_MAP:
            label = FOLDER_MAP[key]
            label_to_imgs[label].extend(_collect_images(p))

    result = {}
    for cls in MEDICAL_CLASSES["lung"]:
        imgs = list(set(label_to_imgs[cls]))  # deduplicate
        if not imgs:
            _log(f"  [WARN] lung/{cls}: no images found in {data_root}")
            _log(f"         Expected sub-folder matching: {list(FOLDER_MAP.keys())}")
        random.shuffle(imgs)
        # IQ-OTH/NCCD is small; take everything if fewer than 50
        result[cls] = imgs[:max(IMAGES_PER_CLASS, len(imgs))]
        _log(f"  lung/{cls:12s}: {len(result[cls])} images collected")
    return result


def _sample_colon(data_root: Path) -> dict[str, list[Path]]:
    """
    Kather 2016 (colorectal-histology-mnist) structure.

    The dataset may have numbered folder names like '01_TUMOR', '02_STROMA', etc.,
    or plain names like 'TUMOR', 'STROMA'. We map all variants to the 8 canonical
    lowercase labels used by config.py CLASS_LABELS['colon'].
    """
    FOLDER_NAME_MAP = {
        # numbered variants
        "01_tumor"  : "tumor",   "tumor"   : "tumor",
        "02_stroma" : "stroma",  "stroma"  : "stroma",
        "03_complex": "complex", "complex" : "complex",
        "04_lympho" : "lympho",  "lympho"  : "lympho",
        "05_debris" : "debris",  "debris"  : "debris",
        "06_mucosa" : "mucosa",  "mucosa"  : "mucosa",
        "07_adipose": "adipose", "adipose" : "adipose",
        "08_empty"  : "empty",   "empty"   : "empty",
        # alternate abbreviated names
        "back"      : "empty",   "norm"    : "mucosa",
        "lym"       : "lympho",  "adi"     : "adipose",
        "tum"       : "tumor",   "str"     : "stroma",
        "deb"       : "debris",  "muc"     : "mucosa",
    }

    label_to_imgs: dict[str, list[Path]] = {c: [] for c in MEDICAL_CLASSES["colon"]}

    for folder in sorted(data_root.rglob("*")):
        if not folder.is_dir():
            continue
        key = folder.name.lower().strip()
        if key in FOLDER_NAME_MAP:
            label = FOLDER_NAME_MAP[key]
            imgs  = (
                list(folder.glob("*.tif")) + list(folder.glob("*.TIF"))
                + _collect_images(folder)
            )
            label_to_imgs[label].extend(imgs)

    result = {}
    for cls in MEDICAL_CLASSES["colon"]:
        imgs = list(set(label_to_imgs[cls]))  # deduplicate paths
        if not imgs:
            _log(f"  [WARN] colon/{cls}: no images found in {data_root}")
            _log("         Check that kmader/colorectal-histology-mnist downloaded correctly.")
        random.shuffle(imgs)
        result[cls] = imgs[:IMAGES_PER_CLASS]
        _log(f"  colon/{cls:12s}: {len(result[cls])} images collected")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic non-medical image generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_non_medical_images(output_dir: Path) -> list[Path]:
    """
    Generate synthetic images that should be REJECTED by the validator model.
    Categories:
      • random_noise  — white-noise images
      • solid_colors  — bright solid-color squares
      • gradients     — linear/radial colour gradients
      • text_images   — images with text (screenshots, documents)
      • patterns      — geometric patterns (checkerboard, stripes)

    These are clearly non-medical and test the OOD rejection gate.
    """
    rng   = np.random.default_rng(SEED)
    paths = []

    def _save(img: Image.Image, subcat: str, idx: int) -> Path:
        dst = output_dir / subcat
        dst.mkdir(parents=True, exist_ok=True)
        path = dst / f"{subcat}_{idx:03d}.jpg"
        img.convert("RGB").save(str(path), quality=90)
        return path

    # 1. Random noise (10 images)
    for i in range(10):
        arr  = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        img  = Image.fromarray(arr)
        paths.append(_save(img, "random_noise", i))

    # 2. Solid vivid colors (5 images)
    vivid_colors = [
        (255, 50,  50),   # red
        (50,  200, 50),   # green
        (50,  50,  255),  # blue
        (255, 200, 50),   # yellow
        (200, 50,  200),  # purple
    ]
    for i, color in enumerate(vivid_colors):
        img = Image.new("RGB", (224, 224), color)
        # Add a simple shape so it's not purely solid
        draw = ImageDraw.Draw(img)
        draw.ellipse([40, 40, 184, 184], fill=tuple(255 - c for c in color))
        paths.append(_save(img, "solid_colors", i))

    # 3. Gradient images (5 images)
    for i in range(5):
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        # Random gradient direction
        angle = rng.integers(0, 4)
        if angle == 0:
            arr[:, :, i % 3] = np.linspace(0, 255, 224, dtype=np.uint8)
        elif angle == 1:
            arr[:, :, i % 3] = np.linspace(255, 0, 224, dtype=np.uint8)[:, None]
        else:
            x, y = np.meshgrid(np.linspace(0, 255, 224), np.linspace(0, 255, 224))
            arr[:, :, 0] = x.astype(np.uint8)
            arr[:, :, 1] = y.astype(np.uint8)
            arr[:, :, 2] = (255 - x).astype(np.uint8)
        paths.append(_save(Image.fromarray(arr), "gradients", i))

    # 4. Text / document-like images (3 images)
    sentences = [
        "This is not a medical scan.",
        "Hello World — testing OOD rejection.",
        "Random text image for validator test.",
    ]
    for i, text in enumerate(sentences):
        img  = Image.new("RGB", (224, 224), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 213, 213], outline=(0, 0, 0), width=2)
        # Draw text in small chunks
        for j, word in enumerate(text.split()):
            draw.text((15, 20 + j * 18), word, fill=(20, 20, 20))
        paths.append(_save(img, "text_images", i))

    # 5. Geometric patterns (2 images)
    for i in range(2):
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        colors_pattern = [
            (255, 100, 0),
            (0, 100, 255),
        ]
        size = 16
        for row in range(0, 224, size):
            for col in range(0, 224, size):
                c = colors_pattern[((row // size) + (col // size)) % 2]
                arr[row:row+size, col:col+size] = c
        paths.append(_save(Image.fromarray(arr), "patterns", i))

    _log(f"  non_medical: {len(paths)} synthetic images generated")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: prepare — download, sample, create zip
# ─────────────────────────────────────────────────────────────────────────────

def prepare_demo_zip(output_zip: str = ZIP_NAME) -> None:
    """
    Download all datasets, sample ≥50 images per class, add synthetic
    non-medical images, and package into a zip file.
    """
    _log("=" * 60)
    _log("PREPARE MODE — generating demo_test_images.zip")
    _log("=" * 60)

    _ensure_kaggle_auth()

    raw_data = Path("./data/demo_raw")
    raw_data.mkdir(parents=True, exist_ok=True)

    # ── Download datasets ────────────────────────────────────────────────────
    _download_dataset("brain", str(raw_data / "brain"))
    _download_dataset("lung",  str(raw_data / "lung"))
    _download_dataset("colon", str(raw_data / "colon"))

    # ── Sample images ────────────────────────────────────────────────────────
    samples: dict[str, dict[str, list[Path]]] = {
        "brain": _sample_brain(raw_data / "brain"),
        "lung" : _sample_lung(raw_data / "lung"),
        "colon": _sample_colon(raw_data / "colon"),
    }

    # ── Build output directory structure ────────────────────────────────────
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)

    manifest = {}

    for cancer_type, class_dict in samples.items():
        manifest[cancer_type] = {}
        for cls, img_paths in class_dict.items():
            dst_dir = DEMO_DIR / cancer_type / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            copied  = 0
            for src_path in img_paths:
                dst_path = dst_dir / src_path.name
                # Avoid name collisions
                if dst_path.exists():
                    dst_path = dst_dir / f"{src_path.stem}_{copied}{src_path.suffix}"
                shutil.copy(str(src_path), dst_path)
                copied += 1
            manifest[cancer_type][cls] = copied
            _log(f"  ✓ {cancer_type}/{cls}: {copied} images")

    # ── Generate non-medical images ──────────────────────────────────────────
    non_med_dir = DEMO_DIR / "non_medical"
    non_med_paths = _generate_non_medical_images(non_med_dir)
    manifest["non_medical"] = {}
    for p in non_med_paths:
        cat = p.parent.name
        manifest["non_medical"][cat] = manifest["non_medical"].get(cat, 0) + 1

    # ── Save manifest ────────────────────────────────────────────────────────
    with open(DEMO_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Create zip ───────────────────────────────────────────────────────────
    _log(f"\nCreating {output_zip} …")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(DEMO_DIR.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(DEMO_DIR.parent))

    zip_size_mb = Path(output_zip).stat().st_size / (1024 * 1024)
    _log(f"Zip created: {output_zip}  ({zip_size_mb:.1f} MB)")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_images = sum(
        v if isinstance(v, int) else sum(v.values())
        for outer in manifest.values()
        for v in (outer.values() if isinstance(outer, dict) else [outer])
    )
    _log("\n" + "=" * 60)
    _log("SUMMARY")
    _log("=" * 60)
    for cancer_type, classes in manifest.items():
        _log(f"  {cancer_type}:")
        for cls, count in classes.items():
            _log(f"    {cls:20s}: {count} images")
    _log(f"\n  TOTAL: {sum(sum(v.values()) for v in manifest.values())} images")
    _log(f"  ZIP  : {output_zip}")

    # ── Auto-download in Colab ────────────────────────────────────────────────
    try:
        from google.colab import files as colab_files  # type: ignore
        _log("\nColab detected — starting download …")
        colab_files.download(output_zip)
    except ImportError:
        _log(f"\nDownload the zip from: {Path(output_zip).resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: api — extract zip, call API, report results
# ─────────────────────────────────────────────────────────────────────────────

def run_api_demo(
    zip_path: str = ZIP_NAME,
    api_url: str  = "http://localhost:8000",
) -> None:
    """
    Extract the demo zip, send every medical image to the /predict endpoint,
    and print a comprehensive results report with per-class metrics.
    """
    try:
        import requests
    except ImportError:
        _log("requests not installed. Run: pip install requests")
        sys.exit(1)

    _log("=" * 60)
    _log("API TEST MODE")
    _log(f"API : {api_url}")
    _log(f"ZIP : {zip_path}")
    _log("=" * 60)

    # ── Health check ─────────────────────────────────────────────────────────
    try:
        health = requests.get(f"{api_url}/health", timeout=10).json()
        _log(f"Server status : {health.get('status')}")
        _log(f"Models loaded : {health.get('models_loaded')}")
        if health.get("models_missing"):
            _log(f"[WARN] Missing models: {health.get('models_missing')}")
    except Exception as exc:
        _log(f"[ERROR] Cannot reach {api_url}/health — is the server running?\n  {exc}")
        sys.exit(1)

    # ── Extract zip ──────────────────────────────────────────────────────────
    extract_dir = Path("./demo_test_images_extracted")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    _log(f"\nExtracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Check if there's an extra wrapper directory
    contents = list(extract_dir.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        image_root = contents[0]
    else:
        image_root = extract_dir

    # ── Run predictions ───────────────────────────────────────────────────────
    results = []

    # Medical images
    for cancer_type, classes in MEDICAL_CLASSES.items():
        for cls in classes:
            class_dir = image_root / cancer_type / cls
            if not class_dir.exists():
                _log(f"[WARN] {cancer_type}/{cls} not found in zip, skipping.")
                continue

            images = sorted(class_dir.glob("*.*"))
            images = [p for p in images
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            _log(f"\n  Testing {cancer_type}/{cls} ({len(images)} images) …")

            for img_path in images:
                result = _call_predict(requests, api_url, img_path, cancer_type)
                result["true_cancer_type"] = cancer_type
                result["true_class"]       = cls
                result["image_path"]       = str(img_path)
                result["is_medical"]       = True
                results.append(result)

    # Non-medical images (should all be rejected)
    non_med_dir = image_root / "non_medical"
    if non_med_dir.exists():
        _log(f"\n  Testing non_medical images (should all be REJECTED) …")
        non_med_images = sorted(non_med_dir.rglob("*.*"))
        non_med_images = [p for p in non_med_images
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        for img_path in non_med_images:
            # Use brain as the cancer_type (doesn't matter — should be rejected)
            result = _call_predict(requests, api_url, img_path, "brain")
            result["true_cancer_type"] = "non_medical"
            result["true_class"]       = img_path.parent.name
            result["image_path"]       = str(img_path)
            result["is_medical"]       = False
            results.append(result)

    # ── Compute & print metrics ───────────────────────────────────────────────
    _print_results_report(results, api_url)

    # ── Save full results JSON ────────────────────────────────────────────────
    results_path = "demo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    _log(f"\nFull results saved to: {results_path}")

    # ── Optional: plots ───────────────────────────────────────────────────────
    try:
        _plot_results(results)
    except ImportError:
        _log("matplotlib/seaborn not available — skipping plots.")


def _call_predict(
    requests_mod,
    api_url: str,
    img_path: Path,
    cancer_type: str,
) -> dict:
    """POST a single image to /predict and return the parsed response."""
    try:
        with open(img_path, "rb") as f:
            resp = requests_mod.post(
                f"{api_url}/predict",
                files={"file": (img_path.name, f, "image/jpeg")},
                data={"cancer_type": cancer_type},
                timeout=30,
            )
        if resp.status_code == 200:
            return resp.json()
        return {
            "status" : "http_error",
            "reason" : f"HTTP {resp.status_code}",
            "message": resp.text[:200],
        }
    except Exception as exc:
        return {"status": "exception", "reason": str(exc), "message": str(exc)}


def _print_results_report(results: list[dict], api_url: str) -> None:
    """Print a structured text summary of all test results."""
    _log("\n" + "=" * 70)
    _log("RESULTS REPORT")
    _log("=" * 70)

    medical   = [r for r in results if r["is_medical"]]
    non_med   = [r for r in results if not r["is_medical"]]

    # ── Non-medical / ambiguous rate ─────────────────────────────────────────
    if non_med:
        # With the confidence-based quality gate, non-medical images show as
        # ambiguous (predicted_class="Invalid Scan") rather than "rejected".
        ambiguous = [r for r in non_med
                     if r.get("prediction_status") == "ambiguous"
                     or r.get("status") == "rejected"]
        rej_rate  = len(ambiguous) / len(non_med) * 100
        _log(f"\n{'─'*60}")
        _log(f"NON-MEDICAL IMAGES  ({len(non_med)} images)")
        _log(f"  Correctly flagged (ambiguous/rejected) : "
             f"{len(ambiguous)} / {len(non_med)}  ({rej_rate:.1f}%)")
        not_flagged = [r for r in non_med if r not in ambiguous]
        if not_flagged:
            _log(f"  [WARN] Not flagged : {len(not_flagged)} images")
            for r in not_flagged[:5]:
                _log(f"         {Path(r['image_path']).name}: status={r.get('status')}"
                     f"  predicted={r.get('predicted_class','–')}")

    # ── Medical classification per cancer type / class ─────────────────────────
    _log(f"\n{'─'*60}")
    _log("MEDICAL IMAGE CLASSIFICATION")

    for cancer_type, classes in MEDICAL_CLASSES.items():
        type_results = [r for r in medical if r["true_cancer_type"] == cancer_type]
        if not type_results:
            continue

        successes = [r for r in type_results if r.get("status") == "success"]
        rejected  = [r for r in type_results if r.get("status") == "rejected"]
        errors    = [r for r in type_results
                     if r.get("status") not in ("success", "rejected")]

        _log(f"\n  [{cancer_type.upper()}]")
        _log(f"  Total     : {len(type_results)}")
        _log(f"  Processed : {len(successes)}  Rejected : {len(rejected)}"
             f"  Errors : {len(errors)}")

        for cls in classes:
            cls_results  = [r for r in successes if r["true_class"] == cls]
            if not cls_results:
                continue

            correct_map  = {
                "brain_glioma"       : "Glioma",
                "brain_meningioma"   : "Meningioma",
                "brain_notumor"      : "No Tumor",
                "brain_pituitary"    : "Pituitary Tumor",
                "lung_benign"        : "Benign Lung Lesion",
                "lung_malignant"     : "Malignant Lung Cancer",
                "lung_normal"        : "Normal Lung",
                "colon_adipose"      : "Adipose Tissue",
                "colon_complex"      : "Complex Glandular Epithelium",
                "colon_debris"       : "Cellular Debris",
                "colon_empty"        : "Background / Empty",
                "colon_lympho"       : "Lymphocytic Infiltrate",
                "colon_mucosa"       : "Normal Mucosa",
                "colon_stroma"       : "Cancer-Associated Stroma",
                "colon_tumor"        : "Colorectal Adenocarcinoma",
            }
            expected = correct_map.get(f"{cancer_type}_{cls}", cls)
            correct  = [r for r in cls_results
                        if r.get("predicted_class") == expected]
            acc      = len(correct) / len(cls_results) * 100 if cls_results else 0

            # Confidence stats
            confs    = [r["confidence"] for r in cls_results if "confidence" in r]
            avg_conf = sum(confs) / len(confs) if confs else 0

            # Prediction status breakdown
            status_counts = {}
            for r in cls_results:
                s = r.get("prediction_status", "unknown")
                status_counts[s] = status_counts.get(s, 0) + 1

            _log(f"    {cls:20s}: {len(cls_results):3d} imgs | "
                 f"acc={acc:5.1f}% | avg_conf={avg_conf:5.1f}% | "
                 f"confident={status_counts.get('confident',0)} "
                 f"low={status_counts.get('low_confidence',0)} "
                 f"ambiguous={status_counts.get('ambiguous',0)}")

    # ── Overall ───────────────────────────────────────────────────────────────
    all_success = [r for r in medical if r.get("status") == "success"]
    correct_map_full = {
        ("brain", "glioma")     : "Glioma",
        ("brain", "meningioma") : "Meningioma",
        ("brain", "notumor")    : "No Tumor",
        ("brain", "pituitary")  : "Pituitary Tumor",
        ("lung",  "benign")     : "Benign Lung Lesion",
        ("lung",  "malignant")  : "Malignant Lung Cancer",
        ("lung",  "normal")     : "Normal Lung",
        ("colon", "adipose")    : "Adipose Tissue",
        ("colon", "complex")    : "Complex Glandular Epithelium",
        ("colon", "debris")     : "Cellular Debris",
        ("colon", "empty")      : "Background / Empty",
        ("colon", "lympho")     : "Lymphocytic Infiltrate",
        ("colon", "mucosa")     : "Normal Mucosa",
        ("colon", "stroma")     : "Cancer-Associated Stroma",
        ("colon", "tumor")      : "Colorectal Adenocarcinoma",
    }
    correct_total = sum(
        1 for r in all_success
        if r.get("predicted_class") == correct_map_full.get(
            (r["true_cancer_type"], r["true_class"]), ""
        )
    )
    overall_acc = correct_total / len(all_success) * 100 if all_success else 0

    _log(f"\n{'─'*60}")
    _log("OVERALL  (medical images)")
    _log(f"  Correct: {correct_total} / {len(all_success)}  ({overall_acc:.1f}%)")
    _log(f"  Total tested: {len(results)}"
         f"  (medical={len(medical)}, non_medical={len(non_med)})")
    _log("=" * 70)


def _plot_results(results: list[dict]) -> None:
    """Generate and save confusion matrix + confidence histogram plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    medical_ok = [r for r in results
                  if r["is_medical"] and r.get("status") == "success"]
    if not medical_ok:
        return

    correct_map = {
        ("brain", "glioma")     : "Glioma",
        ("brain", "meningioma") : "Meningioma",
        ("brain", "notumor")    : "No Tumor",
        ("brain", "pituitary")  : "Pituitary Tumor",
        ("lung",  "benign")     : "Benign Lung Lesion",
        ("lung",  "malignant")  : "Malignant Lung Cancer",
        ("lung",  "normal")     : "Normal Lung",
        ("colon", "adipose")    : "Adipose Tissue",
        ("colon", "complex")    : "Complex Glandular Epithelium",
        ("colon", "debris")     : "Cellular Debris",
        ("colon", "empty")      : "Background / Empty",
        ("colon", "lympho")     : "Lymphocytic Infiltrate",
        ("colon", "mucosa")     : "Normal Mucosa",
        ("colon", "stroma")     : "Cancer-Associated Stroma",
        ("colon", "tumor")      : "Colorectal Adenocarcinoma",
    }

    # ── Per-cancer-type accuracy bar chart ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (cancer_type, classes) in zip(axes, MEDICAL_CLASSES.items()):
        cls_accs = []
        cls_names = []
        for cls in classes:
            rs     = [r for r in medical_ok
                      if r["true_cancer_type"] == cancer_type and r["true_class"] == cls]
            exp    = correct_map.get((cancer_type, cls), "")
            acc    = sum(1 for r in rs if r.get("predicted_class") == exp) / max(len(rs), 1) * 100
            cls_names.append(cls)
            cls_accs.append(acc)
        ax.bar(cls_names, cls_accs, color="steelblue", edgecolor="white")
        ax.set_title(f"{cancer_type.capitalize()} — Per-class Accuracy")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy %")
        ax.axhline(80, color="orange", ls="--", label="80% line")
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Demo API Results — Per-class Accuracy", fontsize=13)
    plt.tight_layout()
    plt.savefig("demo_accuracy_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
    _log("Accuracy chart saved: demo_accuracy_chart.png")

    # ── Confidence distribution ───────────────────────────────────────────────
    confs_correct   = [r["confidence"] for r in medical_ok
                       if r.get("predicted_class") == correct_map.get(
                           (r["true_cancer_type"], r["true_class"]), "")]
    confs_incorrect = [r["confidence"] for r in medical_ok
                       if r.get("predicted_class") != correct_map.get(
                           (r["true_cancer_type"], r["true_class"]), "")]

    plt.figure(figsize=(10, 4))
    if confs_correct:
        plt.hist(confs_correct,   bins=30, alpha=0.75,
                 label=f"Correct ({len(confs_correct)})",   color="steelblue")
    if confs_incorrect:
        plt.hist(confs_incorrect, bins=30, alpha=0.75,
                 label=f"Incorrect ({len(confs_incorrect)})", color="coral")
    plt.axvline(85, color="orange", ls="--", label="Low-conf threshold=85%")
    plt.xlabel("Confidence %")
    plt.ylabel("Count")
    plt.title("Confidence Distribution — Demo API Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig("demo_confidence_hist.png", dpi=150)
    plt.show()
    _log("Confidence histogram saved: demo_confidence_hist.png")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Medical AI Cancer Detection — Demo Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "api"],
        default="prepare",
        help="prepare: download + zip images  |  api: test running FastAPI server",
    )
    parser.add_argument(
        "--zip",
        default=ZIP_NAME,
        help=f"Path to zip file (default: {ZIP_NAME})",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="FastAPI base URL for api mode (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=IMAGES_PER_CLASS,
        help=f"Images to sample per class (default: {IMAGES_PER_CLASS})",
    )

    args = parser.parse_args()

    global IMAGES_PER_CLASS
    IMAGES_PER_CLASS = args.images_per_class

    if args.mode == "prepare":
        prepare_demo_zip(args.zip)
    else:
        run_api_demo(args.zip, args.api_url)


if __name__ == "__main__":
    main()
