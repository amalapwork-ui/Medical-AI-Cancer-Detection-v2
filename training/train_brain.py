# =============================================================================
# Brain Tumor MRI Classification — Google Colab Training Script
# =============================================================================
# Dataset     : Kaggle — "Brain Tumor MRI Dataset" by masoudnickparvar
#               kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
# Classes     : glioma | meningioma | notumor | pituitary
# Architecture: EfficientNetV2S — two-phase transfer learning
#               Phase 1: Train classification head only (backbone frozen)
#               Phase 2: Fine-tune upper backbone layers
# Output      : brain_cancer_effnet.keras  →  place in backend/models/
# =============================================================================
# Run cell-by-cell in Google Colab (each "# %%" is a cell boundary).
# =============================================================================

# %% CELL 1 — Install & Imports
# ─────────────────────────────
# !pip install -q kaggle scikit-learn matplotlib seaborn

import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))


# %% CELL 2 — Kaggle Authentication
# ──────────────────────────────────
from google.colab import files
print("Upload your kaggle.json file:")
files.upload()
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
os.system("cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
print("Kaggle authentication complete.")


# %% CELL 3 — Download & Inspect Dataset
# ────────────────────────────────────────
os.system("kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p ./data/brain --unzip -q")
os.system("find ./data/brain -type d")


# %% CELL 4 — Configuration
# ──────────────────────────
CONFIG = {
    # Paths — adjust if dataset unpacks differently
    "train_dir"     : "./data/brain/Training",
    "test_dir"      : "./data/brain/Testing",
    "output_dir"    : "./output/brain",
    "model_path"    : "./output/brain/brain_cancer_effnet.keras",

    # Model
    "image_size"    : (224, 224),
    "batch_size"    : 32,
    "num_classes"   : 4,
    "classes"       : ["glioma", "meningioma", "notumor", "pituitary"],
    "dropout"       : 0.40,

    # Phase 1 — frozen backbone
    "phase1_epochs" : 15,
    "lr_head"       : 1e-3,

    # Phase 2 — partial backbone unfreeze
    "phase2_epochs" : 60,
    "fine_tune_at"  : 200,   # EfficientNetV2S has ~400 layers; unfreeze top ~200
    "lr_finetune"   : 5e-6,
}
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
print("Config ready. Output dir:", CONFIG["output_dir"])


# %% CELL 5 — Validate Dataset Structure
# ────────────────────────────────────────
def validate_dataset(cfg):
    ok = True
    for split, path in [("Train", cfg["train_dir"]), ("Test", cfg["test_dir"])]:
        p = Path(path)
        if not p.exists():
            print(f"[ERROR] {path} does not exist — check Cell 3 output.")
            ok = False
            continue
        total = 0
        for cls in cfg["classes"]:
            imgs = list((p / cls).glob("*.*"))
            print(f"  [{split}] {cls:15s}: {len(imgs):5d} images")
            total += len(imgs)
        print(f"  [{split}] TOTAL             : {total}\n")
    if not ok:
        raise RuntimeError("Dataset structure invalid. Fix paths above.")

validate_dataset(CONFIG)


# %% CELL 6 — Augmentation Pipeline
# ───────────────────────────────────
# Brain MRI augmentation strategy:
# - Flip + rotation: anatomically plausible
# - Zoom/contrast: simulates different scanner settings
# - NO heavy shear/affine — brain anatomy has some orientation meaning
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.20),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.20),
    layers.RandomBrightness(0.15),
    layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
], name="brain_augmentation")


# %% CELL 7 — Build tf.data Pipelines
# ─────────────────────────────────────
def build_dataset(directory: str, cfg: dict, shuffle: bool, augment: bool):
    """
    EfficientNetV2S uses include_preprocessing=True, so it expects
    raw pixel values in [0, 255] — DO NOT divide by 255 here.
    """
    ds = keras.utils.image_dataset_from_directory(
        directory,
        labels        = "inferred",
        label_mode    = "int",
        class_names   = cfg["classes"],
        image_size    = cfg["image_size"],
        batch_size    = cfg["batch_size"],
        shuffle       = shuffle,
        seed          = SEED,
        interpolation = "bilinear",
    )

    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        if augment:
            images = augmentation(images, training=True)
        return images, labels

    return ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

train_ds = build_dataset(CONFIG["train_dir"], CONFIG, shuffle=True,  augment=True)
val_ds   = build_dataset(CONFIG["test_dir"],  CONFIG, shuffle=False, augment=False)
print("Datasets built.")

# Visualise a sample batch
sample_images, sample_labels = next(iter(train_ds))
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i].numpy().clip(0, 255).astype("uint8"))
    ax.set_title(CONFIG["classes"][sample_labels[i]], fontsize=9)
    ax.axis("off")
plt.suptitle("Augmented Training Samples — Brain MRI")
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/sample_batch.png", dpi=100)
plt.show()


# %% CELL 8 — Compute Class Weights
# ───────────────────────────────────
# NOTE — notumor class imbalance:
# The masoudnickparvar dataset has ~395 notumor training images vs ~826 for each
# tumour class (~2× fewer).  Balanced class weights partially compensate, but the
# model still produces smaller margins on notumor (typically 7–12%) than on
# positive-tumour classes (typically 15–40%).  The inference MARGIN_THRESHOLD in
# config.py is set to 0.07 (not 0.10) to reflect this.  If retraining, consider:
#   • Augmenting notumor images more aggressively (extra flips, brightness jitter)
#   • Oversampling notumor to 800+ images using image_dataset_from_directory repeat
#   • Monitoring per-class recall in the confusion matrix (Cell 12)
def get_class_weights(train_dir: str, classes: list) -> dict:
    labels = []
    for idx, cls in enumerate(classes):
        count = len(list((Path(train_dir) / cls).glob("*.*")))
        labels.extend([idx] * count)
    labels  = np.array(labels)
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    cw      = dict(enumerate(weights))
    for k, v in cw.items():
        print(f"  class {classes[k]:15s}: weight = {v:.4f}")
    return cw

print("Class weights:")
class_weights = get_class_weights(CONFIG["train_dir"], CONFIG["classes"])


# %% CELL 9 — Build Model
# ─────────────────────────
def build_model(cfg: dict, trainable_backbone: bool = False) -> keras.Model:
    """
    EfficientNetV2S backbone + custom classification head.
    include_preprocessing=True: the backbone normalises [0,255] → model-expected range.
    """
    inputs = keras.Input(shape=(*cfg["image_size"], 3), name="input_image")

    backbone = EfficientNetV2S(
        include_top        = False,
        weights            = "imagenet",
        input_tensor       = inputs,
        include_preprocessing = True,
    )
    backbone.trainable = trainable_backbone
    if trainable_backbone:
        # Keep lower feature extraction layers frozen; only fine-tune upper layers
        for layer in backbone.layers[:cfg["fine_tune_at"]]:
            layer.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg["dropout"], name="drop1")(x)
    x = layers.Dense(
        512, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="fc1",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg["dropout"] * 0.5, name="drop2")(x)
    outputs = layers.Dense(cfg["num_classes"], activation="softmax",
                           dtype="float32", name="predictions")(x)

    return keras.Model(inputs, outputs, name="brain_tumor_classifier")


# %% CELL 10 — Phase 1: Train Classification Head
# ─────────────────────────────────────────────────
model = build_model(CONFIG, trainable_backbone=False)
model.summary(line_length=100, expand_nested=False)

model.compile(
    optimizer = keras.optimizers.Adam(CONFIG["lr_head"]),
    loss      = keras.losses.SparseCategoricalCrossentropy(),
    metrics   = ["accuracy"],
)

callbacks_p1 = [
    EarlyStopping("val_accuracy", patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-8, verbose=1),
    ModelCheckpoint(
        f"{CONFIG['output_dir']}/phase1_best.keras",
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
]

print("\n=== PHASE 1: Training head (backbone frozen) ===")
h1 = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = CONFIG["phase1_epochs"],
    callbacks       = callbacks_p1,
    class_weight    = class_weights,
)
print(f"Phase 1 best val_accuracy: {max(h1.history['val_accuracy']):.4f}")


# %% CELL 11 — Phase 2: Fine-tune Backbone
# ──────────────────────────────────────────
# Rebuild with partially unfrozen backbone, then restore Phase 1 weights
tf.keras.backend.clear_session()
model_ft = build_model(CONFIG, trainable_backbone=True)

# Copy Phase 1 head weights into the new model
model_ft.load_weights(f"{CONFIG['output_dir']}/phase1_best.keras")

trainable_params = sum(np.prod(v.shape) for v in model_ft.trainable_weights)
print(f"Trainable parameters in Phase 2: {trainable_params:,}")

model_ft.compile(
    optimizer = keras.optimizers.Adam(CONFIG["lr_finetune"]),
    loss      = keras.losses.SparseCategoricalCrossentropy(),
    metrics   = ["accuracy"],
)

callbacks_p2 = [
    EarlyStopping("val_accuracy", patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau("val_loss", factor=0.3, patience=5, min_lr=1e-9, verbose=1),
    ModelCheckpoint(
        CONFIG["model_path"],
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
]

print("\n=== PHASE 2: Fine-tuning backbone ===")
h2 = model_ft.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = CONFIG["phase1_epochs"] + CONFIG["phase2_epochs"],
    initial_epoch   = len(h1.history["accuracy"]),
    callbacks       = callbacks_p2,
    class_weight    = class_weights,
)
print(f"Phase 2 best val_accuracy: {max(h2.history['val_accuracy']):.4f}")


# %% CELL 12 — Evaluate Best Model
# ──────────────────────────────────
best_model = keras.models.load_model(CONFIG["model_path"])

y_true, y_pred_probs = [], []
for images, labels in val_ds:
    y_pred_probs.extend(best_model.predict(images, verbose=0))
    y_true.extend(labels.numpy())

y_true       = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred       = np.argmax(y_pred_probs, axis=1)
max_probs    = np.max(y_pred_probs, axis=1)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=CONFIG["classes"], digits=4))

# Confusion matrix
cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm,      annot=True, fmt="d",    xticklabels=CONFIG["classes"],
            yticklabels=CONFIG["classes"], cmap="Blues", ax=axes[0])
axes[0].set_title("Counts"); axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")
sns.heatmap(cm_norm, annot=True, fmt=".2%", xticklabels=CONFIG["classes"],
            yticklabels=CONFIG["classes"], cmap="Blues", ax=axes[1])
axes[1].set_title("Normalised"); axes[1].set_ylabel("True"); axes[1].set_xlabel("Predicted")
plt.suptitle("Brain Tumor — Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/confusion_matrix.png", dpi=150)
plt.show()

# Confidence distribution
correct = y_true == y_pred
plt.figure(figsize=(10, 4))
plt.hist(max_probs[correct],  bins=50, alpha=0.75,
         label=f"Correct ({correct.sum()})",    color="steelblue")
plt.hist(max_probs[~correct], bins=50, alpha=0.75,
         label=f"Incorrect ({(~correct).sum()})", color="coral")
plt.axvline(0.85, color="orange", ls="--", label="Confidence=0.85")
plt.xlabel("Max Softmax Probability"); plt.ylabel("Count")
plt.title("Confidence Distribution"); plt.legend()
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/confidence_distribution.png", dpi=150)
plt.show()

# Combined training curves
all_acc      = h1.history["accuracy"]     + h2.history["accuracy"]
all_val_acc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
all_loss     = h1.history["loss"]         + h2.history["loss"]
all_val_loss = h1.history["val_loss"]     + h2.history["val_loss"]
ft_start     = len(h1.history["accuracy"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(all_acc, label="Train"); axes[0].plot(all_val_acc, label="Val")
axes[0].axvline(ft_start, color="r", ls="--", label="Fine-tune start")
axes[0].set_title("Accuracy"); axes[0].set_ylim(0, 1.05); axes[0].legend()
axes[1].plot(all_loss, label="Train"); axes[1].plot(all_val_loss, label="Val")
axes[1].axvline(ft_start, color="r", ls="--", label="Fine-tune start")
axes[1].set_title("Loss"); axes[1].legend()
plt.suptitle("Brain Tumor — Training Curves")
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/training_curves.png", dpi=150)
plt.show()

# Calibration table
print("\nCalibration at confidence thresholds:")
for thr in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]:
    mask = max_probs >= thr
    if mask.sum() == 0:
        continue
    acc_at   = (y_true[mask] == y_pred[mask]).mean()
    coverage = mask.mean()
    print(f"  ≥{thr:.2f}  coverage={coverage:.1%}  accuracy={acc_at:.4f}")


# %% CELL 13 — Save Metadata & Download
# ───────────────────────────────────────
report_dict = classification_report(y_true, y_pred, target_names=CONFIG["classes"], output_dict=True)

metadata = {
    "model_name"       : "brain_cancer_effnet",
    "architecture"     : "EfficientNetV2S",
    "classes"          : CONFIG["classes"],
    "image_size"       : list(CONFIG["image_size"]),
    "preprocessing"    : "include_preprocessing=True — pass raw [0,255] float32",
    "fine_tune_at"     : CONFIG["fine_tune_at"],
    "best_val_accuracy": float(max(h2.history["val_accuracy"])),
    "macro_f1"         : float(report_dict["macro avg"]["f1-score"]),
}
with open(f"{CONFIG['output_dir']}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nMetadata saved:")
print(json.dumps(metadata, indent=2))

from google.colab import files
files.download(CONFIG["model_path"])
print(f"\nDownload started: {CONFIG['model_path']}")
print("Place in: backend/models/brain_cancer_effnet.keras")
