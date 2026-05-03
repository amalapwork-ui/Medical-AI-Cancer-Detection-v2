# =============================================================================
# Lung Cancer CT Classification — Google Colab Training Script  [v3 FIXED]
# =============================================================================
#
# WHAT WAS WRONG IN v2 (and is now fixed)
# ─────────────────────────────────────────
# 1. val_ds was the test set.  EarlyStopping, ReduceLR, and ModelCheckpoint
#    all optimised on the held-out test set — triple data leakage.
#    FIX: two GroupShuffleSplit calls → proper train / val / test.
#
# 2. Phase 2 used clear_session() → rebuild model → load_weights(.keras).
#    After clear_session() the layer name counters reset; subtle suffix
#    mismatches silently load weights into wrong layers.
#    FIX: keep the model object alive; unfreeze the backbone in-place.
#
# 3. Phase 2 epoch calculation was wrong when EarlyStopping fired early.
#    FIX: epochs = initial_epoch + phase2_epochs (not phase1 + phase2).
#
# 4. CLAHE applied during training but NOT in predict.py → train/inference
#    preprocessing mismatch.  FIX: predict.py now applies CLAHE for lung CT
#    (see backend/app/predict.py).  Training script is the source of truth.
#
# 5. from google.colab import files at module level crashed local runs.
#    FIX: guarded behind try/except; script runs locally without changes.
#
# NEW DATASET
# ────────────
# Dataset   : IQ-OTH/NCCD Lung Cancer CT Dataset
#             Kaggle: hamdallak/the-iqothnccd-lung-cancer-dataset
#             Real clinical CT scan slices from Iraq-Oncology Teaching Hospital.
#             110 patients (55 normal, 15 benign, 40 malignant).
#             ~1,190 total images — each patient contributes multiple slices.
#
# Splits (patient-level, no leakage):
#   Train : 68% of patients  (~810 images)
#   Val   : 12% of patients  (~143 images)  ← used by callbacks
#   Test  : 20% of patients  (~237 images)  ← only touched at final eval
#
# Output: backend/models/lung_cancer_final.keras
# Expected: ~82–90% test accuracy (benign recall is the hard part)
# =============================================================================

# %% CELL 1 — Install dependencies
# !pip install -q kaggle scikit-learn matplotlib seaborn opencv-python-headless

import os
import json
import re
import random
import shutil
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Shared training utilities (optional — fall back gracefully if not on path)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.utils.evaluation import plot_training_curves
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# Colab detection — file upload/download helpers only run in Colab
_IN_COLAB = False
try:
    from google.colab import files as _colab_files
    _IN_COLAB = True
except ImportError:
    pass
print(f"Running in Colab: {_IN_COLAB}")


# %% CELL 2 — Kaggle auth (Colab only)
if _IN_COLAB:
    print("Upload kaggle.json:")
    _colab_files.upload()
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    os.system("cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
    print("Kaggle auth done.")


# %% CELL 3 — Download IQ-OTH/NCCD dataset
os.system(
    "kaggle datasets download -d hamdallak/the-iqothnccd-lung-cancer-dataset "
    "-p ./data/iqothnccd --unzip -q"
)
os.system("find ./data/iqothnccd -type d | head -30")


# %% CELL 4 — Discover & map folder structure
# IQ-OTH/NCCD uses inconsistent folder names across releases. Detect them all.

RAW_ROOT  = "./data/iqothnccd"
CLEAN_DIR = "./data/lung_clean"
CLASSES   = ["benign", "malignant", "normal"]

FOLDER_MAP = {
    "benign": "benign", "benign cases": "benign",
    "benign case": "benign", "bengin cases": "benign",
    "malignant": "malignant", "malignant cases": "malignant",
    "malignant case": "malignant",
    "normal": "normal", "normal cases": "normal",
}

def discover_class_folders(root: str) -> dict[str, Path]:
    found = {}
    for p in Path(root).rglob("*"):
        if not p.is_dir():
            continue
        key = p.name.lower().strip()
        if key in FOLDER_MAP:
            label = FOLDER_MAP[key]
            if label not in found:
                found[label] = p
    return found

class_folders = discover_class_folders(RAW_ROOT)
print("Detected class folders:")
for label, path in class_folders.items():
    imgs = list(path.glob("*.*"))
    print(f"  {label:12s}: {len(imgs):4d} images  ← {path}")

missing = [c for c in CLASSES if c not in class_folders]
if missing:
    print(f"\n[WARNING] Classes not found: {missing}")
    print("Check the 'find' output above and update FOLDER_MAP if needed.")


# %% CELL 5 — Build clean flat structure with patient grouping
def infer_patient_id(image_path: Path, class_folder: Path) -> str:
    """
    Extract patient ID from:
      1. Parent sub-directory (image is in class_folder/P001/img.jpg)
      2. Leading numeric prefix in filename (e.g. P001_slice04.jpg)
      3. Fallback: alphabetical batches of ~20 slices per patient
    """
    relative = image_path.relative_to(class_folder)
    parts    = relative.parts
    if len(parts) >= 2:
        return parts[0]
    stem  = image_path.stem
    match = re.match(r'^([a-zA-Z]*\d+)', stem)
    if match:
        return match.group(1)
    all_imgs = sorted(class_folder.glob("*.*"))
    try:
        idx = all_imgs.index(image_path)
        return f"auto_group_{idx // 20:04d}"
    except ValueError:
        return "unknown"

records = []
for class_idx, label in enumerate(CLASSES):
    if label not in class_folders:
        print(f"[SKIP] {label} — folder not found")
        continue
    src_folder = class_folders[label]
    dst_folder = Path(CLEAN_DIR) / label
    dst_folder.mkdir(parents=True, exist_ok=True)
    imgs = sorted(src_folder.rglob("*.*"))
    imgs = [p for p in imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    for img_path in imgs:
        patient_id = infer_patient_id(img_path, src_folder)
        dst_path   = dst_folder / img_path.name
        if not dst_path.exists():
            shutil.copy(str(img_path), dst_path)
        records.append((str(dst_path), class_idx, f"{label}_{patient_id}"))

print(f"\nTotal records collected: {len(records)}")
for cls_idx, label in enumerate(CLASSES):
    count    = sum(1 for r in records if r[1] == cls_idx)
    patients = len({r[2] for r in records if r[1] == cls_idx})
    print(f"  {label:12s}: {count:4d} images, ~{patients} patient groups")


# %% CELL 6 — Three-way patient-level split  (FIXED)
# ─────────────────────────────────────────────────────
# v2 BUG: val_ds aliased test_paths → EarlyStopping and ModelCheckpoint
# both optimised on the test set.  Final evaluation was also the same set.
#
# v3 FIX: two GroupShuffleSplit calls:
#   Step 1 — reserve 20% of PATIENTS as test (held out until final eval only)
#   Step 2 — split remaining 80% into train (85%) / val (15%)
#   Result : train ≈ 68%, val ≈ 12%, test ≈ 20%  (all at patient level)

all_paths  = np.array([r[0] for r in records])
all_labels = np.array([r[1] for r in records])
all_groups = np.array([r[2] for r in records])

# Step 1: carve out test set
gss_test  = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
trainval_idx, test_idx = next(gss_test.split(all_paths, all_labels, all_groups))

# Step 2: split train_val into train and val
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_paths  = all_paths[trainval_idx]
tv_labels = all_labels[trainval_idx]
tv_groups = all_groups[trainval_idx]
train_idx_rel, val_idx_rel = next(gss_val.split(tv_paths, tv_labels, tv_groups))

train_paths  = tv_paths[train_idx_rel]
train_labels = tv_labels[train_idx_rel]
val_paths    = tv_paths[val_idx_rel]
val_labels   = tv_labels[val_idx_rel]
test_paths   = all_paths[test_idx]
test_labels  = all_labels[test_idx]

# Verify zero overlap between all three splits
train_groups = set(all_groups[trainval_idx][train_idx_rel])
val_groups   = set(all_groups[trainval_idx][val_idx_rel])
test_groups  = set(all_groups[test_idx])
assert not (train_groups & val_groups),  "Train/Val group overlap!"
assert not (train_groups & test_groups), "Train/Test group overlap!"
assert not (val_groups   & test_groups), "Val/Test group overlap!"

print(f"Train : {len(train_paths)} images across {len(train_groups)} patient groups")
print(f"Val   : {len(val_paths)}   images across {len(val_groups)} patient groups")
print(f"Test  : {len(test_paths)}  images across {len(test_groups)} patient groups")
print("Patient group overlap between all three splits: 0  ← verified")
for cls_idx, label in enumerate(CLASSES):
    tr = (train_labels == cls_idx).sum()
    vl = (val_labels   == cls_idx).sum()
    te = (test_labels  == cls_idx).sum()
    print(f"  {label:12s}: {tr} train / {vl} val / {te} test")


# %% CELL 7 — CLAHE preprocessing
# ──────────────────────────────────────────────────────────────────────────────
# CT scan slices from different scanners have very different brightness/contrast.
# CLAHE (Contrast Limited Adaptive Histogram Equalization) normalises local
# contrast before the model sees the image.
#
# IMPORTANT: predict.py must apply identical CLAHE before passing images to
# the lung model. See backend/app/predict.py → _apply_clahe().
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE = (224, 224)

def apply_clahe(img_array: np.ndarray) -> np.ndarray:
    """Apply CLAHE per channel. Input/output: uint8 (H, W, 3)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = [clahe.apply(img_array[:, :, c]) for c in range(3)]
    return np.stack(channels, axis=-1)

def load_and_preprocess(path: bytes, label: int):
    """tf.py_function wrapper: load → RGB → CLAHE → resize → float32."""
    path_str = path.numpy().decode("utf-8")
    img      = cv2.imread(path_str)
    if img is None:
        img = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_clahe(img)
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32), label

def tf_load(path, label):
    img, lbl = tf.py_function(
        load_and_preprocess, [path, label], [tf.float32, tf.int64]
    )
    img.set_shape((*IMAGE_SIZE, 3))
    lbl.set_shape(())
    return img, lbl


# %% CELL 8 — Configuration
CONFIG = {
    "output_dir"    : "./output/lung",
    "model_path"    : "./output/lung/lung_cancer_final.keras",
    "image_size"    : IMAGE_SIZE,
    "batch_size"    : 16,
    "num_classes"   : 3,
    "classes"       : CLASSES,
    "dropout"       : 0.50,
    "l2_reg"        : 5e-4,
    "phase1_epochs" : 20,
    "lr_head"       : 5e-4,
    "phase2_epochs" : 80,
    "fine_tune_at"  : 250,
    "lr_finetune"   : 2e-6,
    "focal_gamma"   : 2.0,
}
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)


# %% CELL 9 — Augmentation (heavier for small dataset)
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.20),
    layers.RandomTranslation(0.10, 0.10),
    layers.GaussianNoise(0.015),
], name="lung_aug")


# %% CELL 10 — tf.data pipelines
BATCH   = CONFIG["batch_size"]
AUTOTUNE = tf.data.AUTOTUNE

def build_dataset(paths, labels, shuffle, augment):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels.astype(np.int64)))
    ds = ds.map(tf_load, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.batch(BATCH)
    if augment:
        ds = ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    return ds.prefetch(AUTOTUNE)

train_ds = build_dataset(train_paths, train_labels, shuffle=True,  augment=True)
val_ds   = build_dataset(val_paths,   val_labels,   shuffle=False, augment=False)
test_ds  = build_dataset(test_paths,  test_labels,  shuffle=False, augment=False)
print("Datasets ready.")

# Sample visualisation
imgs, lbls = next(iter(train_ds))
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i].numpy().clip(0, 255).astype("uint8"))
    ax.set_title(CLASSES[lbls[i].numpy()], fontsize=9)
    ax.axis("off")
plt.suptitle("CLAHE-enhanced CT slices (training, augmented)")
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/sample_batch.png", dpi=100)
plt.show()


# %% CELL 11 — Class weights (computed on TRAIN set only)
weights_arr  = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(weights_arr))
print("Class weights (computed on train set only):")
for k, v in class_weights.items():
    print(f"  {CLASSES[k]:12s}: {v:.4f}")


# %% CELL 12 — Focal Loss (alpha-weighted)
# ──────────────────────────────────────────────────────────────────────────────
# SparseFocalLoss with alpha (class weights) handles two problems at once:
#   γ (gamma) — down-weights easy examples, focuses on hard misclassifications
#   α (alpha) — per-class weight corrects for the small benign population
#
# NOTE: class_weights are baked INTO the loss object here, so do NOT also
# pass class_weight= to model.fit() — that would double-count them.
# ──────────────────────────────────────────────────────────────────────────────
class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, class_weights: dict = None, **kw):
        super().__init__(**kw)
        self.gamma    = gamma
        self._class_w = (
            tf.constant(
                [class_weights[i] for i in range(len(class_weights))],
                dtype=tf.float32,
            ) if class_weights else None
        )

    def call(self, y_true, y_pred):
        y_true   = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred   = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0)
        batch_sz = tf.shape(y_true)[0]
        indices  = tf.stack([tf.range(batch_sz), y_true], axis=1)
        pt       = tf.gather_nd(y_pred, indices)
        ce       = -tf.math.log(pt)
        fw       = tf.pow(1.0 - pt, self.gamma)
        if self._class_w is not None:
            alpha = tf.gather(self._class_w, y_true)
            loss  = alpha * fw * ce
        else:
            loss  = fw * ce
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma})
        return cfg


# %% CELL 13 — Model builder
# Returns (model, backbone) so Phase 2 can unfreeze backbone in-place.
def build_model(cfg) -> tuple[keras.Model, keras.Model]:
    inputs   = keras.Input(shape=(*cfg["image_size"], 3), name="input_ct")
    backbone = EfficientNetV2S(
        include_top           = False,
        weights               = "imagenet",
        input_tensor          = inputs,
        include_preprocessing = True,   # expects raw [0,255] float32
    )
    backbone.trainable = False  # Phase 1: frozen

    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg["dropout"])(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(cfg["l2_reg"]),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg["dropout"] * 0.5)(x)
    outputs = layers.Dense(
        cfg["num_classes"], activation="softmax", dtype="float32", name="predictions"
    )(x)
    model = keras.Model(inputs, outputs, name="lung_cancer_classifier")
    return model, backbone


# %% CELL 14 — Phase 1: Head training (backbone frozen)
model, backbone = build_model(CONFIG)
model.summary(line_length=100)

focal_loss_p1 = SparseFocalLoss(gamma=CONFIG["focal_gamma"], class_weights=class_weights)
model.compile(
    optimizer = keras.optimizers.Adam(CONFIG["lr_head"]),
    loss      = focal_loss_p1,
    metrics   = ["accuracy"],
)

cbs_p1 = [
    EarlyStopping("val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau("val_loss", factor=0.5, patience=4, min_lr=1e-8, verbose=1),
    ModelCheckpoint(
        f"{CONFIG['output_dir']}/phase1_best.keras",
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
]

print("\n=== PHASE 1: Head training (backbone frozen) ===")
h1 = model.fit(
    train_ds,
    validation_data = val_ds,     # ← true held-out val set (not test!)
    epochs          = CONFIG["phase1_epochs"],
    callbacks       = cbs_p1,
)
p1_epochs_run = len(h1.history["accuracy"])
print(f"Phase 1 ran {p1_epochs_run} epochs. Best val_accuracy: {max(h1.history['val_accuracy']):.4f}")


# %% CELL 15 — Phase 2: In-place backbone fine-tuning  (FIXED)
# ─────────────────────────────────────────────────────────────────────────────
# v2 BUG: clear_session() + rebuild + load_weights(.keras) — after
#   clear_session() layer name counters reset; suffix mismatches silently
#   loaded weights into wrong layers.
#
# v3 FIX: unfreeze the backbone object we already hold a reference to.
#   No clear_session, no weight reload, no layer name ambiguity.
# ─────────────────────────────────────────────────────────────────────────────
backbone.trainable = True
for layer in backbone.layers[:CONFIG["fine_tune_at"]]:
    layer.trainable = False

trainable_params = sum(np.prod(v.shape) for v in model.trainable_weights)
print(f"Trainable parameters in Phase 2: {trainable_params:,}")

focal_loss_p2 = SparseFocalLoss(gamma=CONFIG["focal_gamma"], class_weights=class_weights)
model.compile(
    optimizer = keras.optimizers.Adam(CONFIG["lr_finetune"]),
    loss      = focal_loss_p2,
    metrics   = ["accuracy"],
)

p2_total_epochs = p1_epochs_run + CONFIG["phase2_epochs"]  # correct calculation

cbs_p2 = [
    EarlyStopping("val_accuracy", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau("val_loss", factor=0.3, patience=6, min_lr=1e-9, verbose=1),
    ModelCheckpoint(
        CONFIG["model_path"],
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
]

print("\n=== PHASE 2: Backbone fine-tuning (in-place, no weight reload) ===")
h2 = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = p2_total_epochs,
    initial_epoch   = p1_epochs_run,
    callbacks       = cbs_p2,
)
print(f"Phase 2 best val_accuracy: {max(h2.history['val_accuracy']):.4f}")


# %% CELL 16 — Test-Time Augmentation (TTA) evaluation on held-out TEST set
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: test_ds was never seen during training or used by any callback.
# This is the first and only time we evaluate on it.
# ─────────────────────────────────────────────────────────────────────────────
best_model = keras.models.load_model(CONFIG["model_path"])

TTA_STEPS = 8
tta_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.15),
], name="tta_aug")

print(f"\nRunning TTA ({TTA_STEPS}×) on held-out test set …")
y_true_list, tta_probs_list = [], []

for images, labels in test_ds:
    y_true_list.extend(labels.numpy())
    batch_tta = np.zeros((images.shape[0], CONFIG["num_classes"]), dtype=np.float32)
    for _ in range(TTA_STEPS):
        aug_imgs  = tta_aug(images, training=True)
        batch_tta += best_model.predict(aug_imgs, verbose=0)
    tta_probs_list.extend(batch_tta / TTA_STEPS)

y_true       = np.array(y_true_list)
y_pred_probs = np.array(tta_probs_list)
y_pred       = np.argmax(y_pred_probs, axis=1)
max_probs    = np.max(y_pred_probs, axis=1)

print("\n=== CLASSIFICATION REPORT (TTA, held-out test set) ===")
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
print("NOTE: Benign recall is expected to be lowest — smallest class.")

# OvR AUC
try:
    y_bin = label_binarize(y_true, classes=list(range(CONFIG["num_classes"])))
    auc   = roc_auc_score(y_bin, y_pred_probs, average="macro", multi_class="ovr")
    print(f"Macro OvR AUC: {auc:.4f}")
except Exception as e:
    print(f"AUC skipped: {e}")
    auc = None

# Confusion matrix
cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm,      annot=True, fmt="d",    xticklabels=CLASSES,
            yticklabels=CLASSES, cmap="Blues", ax=axes[0])
axes[0].set_title("Counts")
sns.heatmap(cm_norm, annot=True, fmt=".2%", xticklabels=CLASSES,
            yticklabels=CLASSES, cmap="Blues", ax=axes[1])
axes[1].set_title("Normalised")
for ax in axes:
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
plt.suptitle("Lung Cancer (IQ-OTH/NCCD) — Confusion Matrix (TTA, test set)")
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/confusion_matrix.png", dpi=150)
plt.show()

# Calibration at various confidence thresholds
print("\nCalibration at confidence thresholds (test set):")
for thr in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
    mask = max_probs >= thr
    if not mask.any():
        continue
    acc_at   = (y_true[mask] == y_pred[mask]).mean()
    coverage = mask.mean()
    print(f"  ≥{thr:.2f}  coverage={coverage:.1%}  accuracy={acc_at:.4f}")

# Training curves across both phases
if _HAS_UTILS:
    plot_training_curves(h1, h2, CONFIG["output_dir"],
                         fine_tune_start=p1_epochs_run)
else:
    all_acc     = h1.history["accuracy"]     + h2.history["accuracy"]
    all_val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(all_acc,     label="Train")
    ax.plot(all_val_acc, label="Val")
    ax.axvline(p1_epochs_run, color="r", ls="--", label="Fine-tune start")
    ax.set_title("Lung Cancer — Training Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/training_curves.png", dpi=150)
    plt.show()


# %% CELL 17 — Save metadata
report_dict = classification_report(
    y_true, y_pred, target_names=CLASSES, output_dict=True
)
metadata = {
    "model_name"        : "lung_cancer_final",
    "architecture"      : "EfficientNetV2S",
    "dataset"           : "IQ-OTH/NCCD (hamdallak/the-iqothnccd-lung-cancer-dataset)",
    "classes"           : CLASSES,
    "image_size"        : list(CONFIG["image_size"]),
    "preprocessing"     : "CLAHE (clipLimit=2.0, tileGridSize=8×8) → float32 [0,255]",
    "splitting"         : "Patient-level GroupShuffleSplit: 68/12/20 train/val/test",
    "loss"              : f"SparseFocalLoss gamma={CONFIG['focal_gamma']} + alpha class weights",
    "tta_steps"         : TTA_STEPS,
    "phase1_epochs_run" : p1_epochs_run,
    "best_val_accuracy" : float(max(h2.history["val_accuracy"])),
    "test_accuracy"     : float((y_true == y_pred).mean()),
    "macro_f1"          : float(report_dict["macro avg"]["f1-score"]),
    "macro_auc"         : float(auc) if auc is not None else None,
    "inference_note"    : (
        "predict.py must apply identical CLAHE before passing lung CT images "
        "to this model. See backend/app/predict.py → _apply_clahe()."
    ),
}
with open(f"{CONFIG['output_dir']}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(json.dumps(metadata, indent=2))

# %% CELL 18 — Download (Colab only)
if _IN_COLAB:
    _colab_files.download(CONFIG["model_path"])
    print("\nPlace in: backend/models/lung_cancer_final.keras")
    print("Verify in config.py: CLASS_LABELS['lung'] =", CLASSES)
else:
    print(f"\nModel saved to: {CONFIG['model_path']}")
    print("Copy to: backend/models/lung_cancer_final.keras")
