"""
STEP 2 — Grad-CAM Heatmap Visualisation
=========================================
Shows WHICH brain regions the model focused on to decide normal vs abnormal.
Run this AFTER step1_train_model.py has finished.

HOW TO RUN:
    python backend/step2_gradcam.py

Output: results/gradcam_samples.png
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
MODEL_PATH   = BASE_DIR / "results" / "best_model.h5"
DATASET_DIR  = BASE_DIR / "dataset"
RESULTS_DIR  = BASE_DIR / "results"
IMG_SIZE     = (224, 224)
NUM_SAMPLES  = 6      # number of images to visualize (3 normal + 3 abnormal)
RANDOM_SEED  = 42
random.seed(RANDOM_SEED)


# ─── Load model ────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# Build Grad-CAM model using last conv layer of VGG16
last_conv_layer = model.get_layer("block5_conv3")
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)
print("Grad-CAM model ready.\n")


# ─── Load test CSV ─────────────────────────────
def load_test_data():
    folder   = DATASET_DIR / "test"
    csv_path = folder / "_classes.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["filepath"] = df["filename"].apply(lambda f: str(folder / f.strip()))
    df["label"]    = df["normal"].apply(lambda x: "normal" if x == 1 else "abnormal")
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    return df

test_df = load_test_data()
print(f"Test set loaded: {len(test_df)} images")
print(f"  Normal   : {(test_df['label']=='normal').sum()}")
print(f"  Abnormal : {(test_df['label']=='abnormal').sum()}\n")

# Pick balanced sample: 3 normal + 3 abnormal
normal_samples   = test_df[test_df["label"] == "normal"].sample(
    min(NUM_SAMPLES // 2, (test_df["label"] == "normal").sum()),
    random_state=RANDOM_SEED
)
abnormal_samples = test_df[test_df["label"] == "abnormal"].sample(
    min(NUM_SAMPLES // 2, (test_df["label"] == "abnormal").sum()),
    random_state=RANDOM_SEED
)
samples = pd.concat([normal_samples, abnormal_samples]).reset_index(drop=True)
print(f"Generating Grad-CAM for {len(samples)} sample images...\n")


# ─── Grad-CAM computation ──────────────────────
def compute_gradcam(img_array):
    """Returns heatmap and prediction confidence for one image (1,224,224,3)."""
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        abnormal_prob = predictions[:, 0]
        predicted_abnormal = abnormal_prob[0] > 0.5
        class_channel = abnormal_prob if predicted_abnormal else (1 - abnormal_prob)

    grads       = tape.gradient(class_channel, conv_outputs)
    pooled_grads= tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out    = conv_outputs[0]
    heatmap     = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(abnormal_prob[0])


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    """Overlays Grad-CAM heatmap on original image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return img, superimposed


# ─── Generate plots ────────────────────────────
fig, axes = plt.subplots(len(samples), 3, figsize=(13, len(samples) * 4))
fig.suptitle("Grad-CAM — Fetal Brain Abnormality Detection\n(Red/yellow = regions the model focused on)",
             fontsize=12, y=1.01)

for i, row in samples.iterrows():
    img_path   = row["filepath"]
    true_label = row["label"]

    # Preprocess image
    img_arr = load_img(img_path, target_size=IMG_SIZE)
    img_arr = img_to_array(img_arr) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Get Grad-CAM
    heatmap, abnormal_prob = compute_gradcam(img_arr)

    # abnormal_prob > 0.5 → predicted abnormal
    pred_label = "abnormal" if abnormal_prob > 0.5 else "normal"
    confidence = abnormal_prob if pred_label == "abnormal" else (1 - abnormal_prob)
    correct    = "✓ Correct" if pred_label == true_label else "✗ Wrong"

    original, overlaid = overlay_heatmap(img_path, heatmap)

    ax_row = axes[i]
    ax_row[0].imshow(original)
    ax_row[0].set_title(f"Original\nTrue: {true_label}", fontsize=9)

    ax_row[1].imshow(heatmap, cmap="jet")
    ax_row[1].set_title("Grad-CAM heatmap\n(activation regions)", fontsize=9)

    ax_row[2].imshow(overlaid)
    ax_row[2].set_title(
        f"Overlay\nPred: {pred_label} ({confidence:.0%}) — {correct}",
        fontsize=9,
        color="green" if "Correct" in correct else "red"
    )

    for ax in ax_row:
        ax.axis("off")

    print(f"[{i+1}] {os.path.basename(img_path)[:40]}")
    print(f"     True: {true_label} | Pred: {pred_label} ({confidence:.0%}) | {correct}")

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "gradcam_samples.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved: {out_path}")
print("\n" + "="*55)
print("PROJECT COMPLETE! Your results folder contains:")
print("  best_model.h5              trained VGG16 model")
print("  training_curves.png        accuracy & loss plots")
print("  classification_report.txt  precision, recall, F1")
print("  confusion_matrix.png       prediction breakdown")
print("  gradcam_samples.png        explainability heatmaps")
print("="*55)
