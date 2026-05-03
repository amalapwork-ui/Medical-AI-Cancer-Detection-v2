"""
Standardized augmentation strategies for all medical image models.

Design decisions:
- All models share IMAGE_SIZE = (224, 224) to enable consistent preprocessing.
- EfficientNetV2S with include_preprocessing=True expects raw [0, 255] float32 — do NOT
normalize to [0,1] before passing to the model; the backbone handles it internally.
- "heavy" augmentation suits histopathology (colon) where rotation/flip is always valid.
- "medium" suits brain MRI / lung CT where orientation carries some clinical meaning.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def get_augmentation(severity: str = "medium") -> keras.Sequential:
    """
    Build a Keras Sequential augmentation pipeline.

    severity:
        "light"  — flips + small rotation only (val/test-time TTA)
        "medium" — standard medical imaging augmentation
        "heavy"  — max augmentation for histopathology
    """
    if severity == "light":
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.05),
        ], name="aug_light")

    if severity == "medium":
        return keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ], name="aug_medium")

    if severity == "heavy":
        return keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.3),
            layers.RandomBrightness(0.2),
            layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
            layers.GaussianNoise(0.015),
        ], name="aug_heavy")

    raise ValueError(f"Unknown severity '{severity}'. Choose: light | medium | heavy")


def apply_mixup(images: tf.Tensor, labels_onehot: tf.Tensor, alpha: float = 0.3):
    """
    MixUp data augmentation (Zhang et al., 2018).

    Blends pairs of images and their one-hot labels using a Beta(alpha, alpha)
    mixing coefficient. This significantly reduces overconfident softmax outputs,
    which is the root cause of random images being classified with high confidence.

    Args:
        images:       float32 tensor (B, H, W, C)
        labels_onehot: float32 tensor (B, num_classes)
        alpha:        Beta distribution parameter; 0.2–0.4 works well

    Returns:
        mixed_images, mixed_labels (same shapes as inputs)
    """
    batch_size = tf.shape(images)[0]

    # Sample lambda from Beta(alpha, alpha) via a stick-breaking approximation
    lam = tf.random.uniform(shape=(batch_size,), minval=0.0, maxval=1.0)
    lam = tf.where(lam < alpha, lam + alpha, lam)
    lam = tf.clip_by_value(lam, 0.0, 1.0)

    lam_img   = tf.reshape(lam, [-1, 1, 1, 1])
    lam_label = tf.reshape(lam, [-1, 1])

    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, indices)
    labels_shuffled = tf.gather(labels_onehot, indices)

    mixed_images = lam_img * images + (1.0 - lam_img) * images_shuffled
    mixed_labels = lam_label * labels_onehot + (1.0 - lam_label) * labels_shuffled

    return mixed_images, mixed_labels
