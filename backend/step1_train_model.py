"""
STEP 1 — VGG16 Transfer Learning: Fetal Brain Abnormality Detection
=====================================================================
Dataset structure expected:
    dataset/
        train/   _classes.csv + .jpg images
        valid/   _classes.csv + .jpg images
        test/    _classes.csv + .jpg images

HOW TO RUN:
    pip install -r backend/requirements.txt
    python backend/step1_train_model.py

Outputs saved in results/ folder:
    best_model.h5              best model weights
    training_curves.png        accuracy & loss plots
    classification_report.txt  precision, recall, F1
    confusion_matrix.png       visual confusion matrix
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATASET_DIR  = BASE_DIR / "dataset"
RESULTS_DIR  = BASE_DIR / "results"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
EPOCHS_P1    = 15       # Phase 1: train head only (base frozen)
EPOCHS_P2    = 10       # Phase 2: fine-tune last VGG layers
RANDOM_SEED  = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# STEP A — Load CSVs and convert to normal/abnormal
# The dataset has 16 classes. We merge all
# abnormality columns into one "abnormal" label.
# ─────────────────────────────────────────────
def load_split(split_name):
    folder = DATASET_DIR / split_name
    csv_path = folder / "_classes.csv"

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()          # remove spaces from column names

    # Build full image path
    df["filepath"] = df["filename"].apply(lambda f: str(folder / f.strip()))

    # label = "normal" if normal==1, else "abnormal"
    df["label"] = df["normal"].apply(lambda x: "normal" if x == 1 else "abnormal")

    # Keep only filepath and label
    df = df[["filepath", "label"]].reset_index(drop=True)

    # Remove rows where image file doesn't exist
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)

    return df

print("Loading dataset splits...")
train_df = load_split("train")
valid_df = load_split("valid")
test_df  = load_split("test")

print(f"Train : {len(train_df)} images  {train_df['label'].value_counts().to_dict()}")
print(f"Valid : {len(valid_df)} images  {valid_df['label'].value_counts().to_dict()}")
print(f"Test  : {len(test_df)} images   {test_df['label'].value_counts().to_dict()}")


# ─────────────────────────────────────────────
# STEP B — Image Data Generators
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.9, 1.1]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["abnormal", "normal"],   # abnormal=0, normal=1
    shuffle=True,
    seed=RANDOM_SEED
)

valid_gen = val_test_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["abnormal", "normal"],
    shuffle=False
)

test_gen = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["abnormal", "normal"],
    shuffle=False
)

print(f"\nClass indices: {train_gen.class_indices}")
# abnormal = 0, normal = 1


# ─────────────────────────────────────────────
# STEP C — Build VGG16 Model
# ─────────────────────────────────────────────
def build_vgg16_model():
    # Load VGG16 pretrained on ImageNet, remove top classifier
    base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False      # freeze all base layers for Phase 1

    # Add custom classification head
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)   # binary output

    model = Model(inputs=base.input, outputs=output)
    return model, base

model, base_model = build_vgg16_model()
print(f"\nModel built — Total layers: {len(model.layers)}")
print(f"Trainable layers (Phase 1): {sum(1 for l in model.layers if l.trainable)}")


# ─────────────────────────────────────────────
# STEP D — Phase 1: Train custom head only
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("PHASE 1 — Training custom head (VGG16 base frozen)")
print("="*55)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks_p1 = [
    ModelCheckpoint(
        os.path.join(RESULTS_DIR, "best_model_phase1.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    )
]

history1 = model.fit(
    train_gen,
    epochs=EPOCHS_P1,
    validation_data=valid_gen,
    callbacks=callbacks_p1,
    verbose=1
)


# ─────────────────────────────────────────────
# STEP E — Phase 2: Fine-tune last 4 VGG16 layers
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("PHASE 2 — Fine-tuning last 4 layers of VGG16")
print("="*55)

# Unfreeze last 4 layers of VGG16
for layer in base_model.layers[-4:]:
    layer.trainable = True

print(f"Trainable layers (Phase 2): {sum(1 for l in model.layers if l.trainable)}")

# Use a very small LR to avoid destroying pretrained weights
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks_p2 = [
    ModelCheckpoint(
        os.path.join(RESULTS_DIR, "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_P2,
    validation_data=valid_gen,
    callbacks=callbacks_p2,
    verbose=1
)


# ─────────────────────────────────────────────
# STEP F — Plot Training Curves
# ─────────────────────────────────────────────
def plot_training(h1, h2):
    acc   = h1.history["accuracy"]     + h2.history["accuracy"]
    val   = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]         + h2.history["loss"]
    vloss = h1.history["val_loss"]     + h2.history["val_loss"]
    sep   = len(h1.history["accuracy"])   # where Phase 2 starts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VGG16 Transfer Learning — Fetal Brain Abnormality Detection", fontsize=13)

    ax1.plot(acc,  label="Train accuracy",  color="steelblue")
    ax1.plot(val,  label="Val accuracy",    color="coral")
    ax1.axvline(sep - 1, color="gray", linestyle="--", linewidth=1, label="Fine-tune start")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(loss,  label="Train loss", color="steelblue")
    ax2.plot(vloss, label="Val loss",   color="coral")
    ax2.axvline(sep - 1, color="gray", linestyle="--", linewidth=1, label="Fine-tune start")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

plot_training(history1, history2)


# ─────────────────────────────────────────────
# STEP G — Evaluate on Test Set
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("TEST SET EVALUATION")
print("="*55)

test_gen.reset()
preds     = model.predict(test_gen, verbose=1)
y_pred    = (preds > 0.5).astype(int).flatten()
y_true    = test_gen.classes
# class_indices: abnormal=0, normal=1
class_names = ["abnormal", "normal"]

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Test accuracy
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"Test Loss     : {test_loss:.4f}")

# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix — Test Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"Saved: {cm_path}")

print("\n" + "="*55)
print("All results saved in results/ folder:")
print("  best_model.h5")
print("  training_curves.png")
print("  classification_report.txt")
print("  confusion_matrix.png")
print("\nNext → run step2_gradcam.py")
print("="*55)
