"""
STEP 3 — Fetal Brain MRI Abnormality Detector — Streamlit Web App
===================================================================
HOW TO RUN:
    pip install streamlit
    streamlit run step3_app.py

Then open http://localhost:8501 in your browser.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from backend.mri_validation import load_mri_reference_profile, validate_mri_like_image
from PIL import Image
import streamlit as st
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fetal Brain MRI Analyzer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #58a6ff;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #8b949e;
    font-size: 0.95rem;
    font-weight: 300;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed #30363d;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    background: #161b22;
    transition: border-color 0.3s;
    margin: 1rem 0;
}
.upload-zone:hover { border-color: #58a6ff; }

/* Result cards */
.result-card {
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin: 1rem 0;
    border: 1px solid;
}
.result-normal {
    background: linear-gradient(135deg, #0d2818 0%, #0d1117 100%);
    border-color: #2ea043;
}
.result-abnormal {
    background: linear-gradient(135deg, #2d1117 0%, #0d1117 100%);
    border-color: #f85149;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-normal .result-title  { color: #3fb950; }
.result-abnormal .result-title { color: #f85149; }
.result-subtitle { color: #8b949e; font-size: 0.88rem; }

/* Confidence bar */
.conf-bar-wrap {
    margin: 1.2rem 0 0.4rem;
    background: #21262d;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
.conf-bar-normal   { background: linear-gradient(90deg, #2ea043, #3fb950); }
.conf-bar-abnormal { background: linear-gradient(90deg, #da3633, #f85149); }

/* Info box */
.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 1.5rem;
}
.info-box strong { color: #e6edf3; }

/* Disclaimer */
.disclaimer {
    text-align: center;
    font-size: 0.78rem;
    color: #484f58;
    margin-top: 2.5rem;
    padding: 1rem;
    border-top: 1px solid #21262d;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stSpinner > div { border-top-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
MODEL_PATH = "results/best_model.h5"
IMG_SIZE   = (224, 224)

@st.cache_resource
def load_vgg_model():
    if not os.path.exists(MODEL_PATH):
        return None
    load_mri_reference_profile()
    model = load_model(MODEL_PATH)
    # Build Grad-CAM sub-model
    last_conv  = model.get_layer("block5_conv3")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output]
    )
    return model, grad_model

result = load_vgg_model()


# ─────────────────────────────────────────────
# GRAD-CAM HELPERS
# ─────────────────────────────────────────────
def compute_gradcam(grad_model, img_array):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        abnormal_prob = predictions[:, 0]
        predicted_abnormal = abnormal_prob[0] > 0.5
        class_channel = abnormal_prob if predicted_abnormal else (1 - abnormal_prob)
    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_outputs[0]
    heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(abnormal_prob[0])


def overlay_heatmap(pil_image, heatmap, alpha=0.45):
    img = np.array(pil_image.resize(IMG_SIZE))
    if img.ndim == 2:                        # grayscale → RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:                  # RGBA → RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    hm_resized  = cv2.resize(heatmap, IMG_SIZE)
    hm_uint8    = np.uint8(255 * hm_resized)
    hm_colored  = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_colored  = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 1 - alpha, hm_colored, alpha, 0)
    return img, superimposed


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    return Image.open(buf)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧠 Fetal Brain MRI Analyzer</h1>
  <p>VGG16 Transfer Learning · Grad-CAM Explainability · Binary Classification</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL STATUS
# ─────────────────────────────────────────────
if result is None:
    st.error("⚠️  Model not found at `results/best_model.h5`. "
             "Please run `step1_train_model.py` first.")
    st.stop()

model, grad_model = result
st.success("✅  Model loaded successfully — VGG16 fine-tuned on fetal brain MRI dataset")

# ─────────────────────────────────────────────
# FILE UPLOADER
# ─────────────────────────────────────────────
st.markdown("### Upload an MRI Scan")
uploaded = st.file_uploader(
    "Drag and drop or click to browse",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown("""
    <div class="upload-zone">
        <p style="font-size:2rem; margin-bottom:0.4rem">📂</p>
        <p style="color:#8b949e; margin:0">Upload a fetal brain MRI image (JPG / PNG)</p>
        <p style="color:#484f58; font-size:0.8rem; margin-top:0.4rem">Supports grayscale and RGB scans</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
pil_image = Image.open(uploaded).convert("RGB")
is_mri_like, rejection_reason = validate_mri_like_image(pil_image)
if not is_mri_like:
    st.error(rejection_reason + " Please upload a valid fetal brain MRI image.")
    st.stop()

with st.spinner("Analysing scan…"):
    # Preprocess
    img_resized = pil_image.resize(IMG_SIZE)
    img_array   = img_to_array(img_resized) / 255.0
    img_array   = np.expand_dims(img_array, axis=0)

    # Predict + Grad-CAM
    heatmap, abnormal_prob = compute_gradcam(grad_model, img_array)
    pred_label  = "abnormal" if abnormal_prob > 0.5 else "normal"
    confidence  = abnormal_prob if pred_label == "abnormal" else (1 - abnormal_prob)
    conf_pct    = confidence * 100

# ─────────────────────────────────────────────
# RESULT CARD
# ─────────────────────────────────────────────
card_cls  = "result-abnormal" if pred_label == "abnormal" else "result-normal"
bar_cls   = "conf-bar-abnormal" if pred_label == "abnormal" else "conf-bar-normal"
icon      = "⚠️" if pred_label == "abnormal" else "✅"
title_txt = "Abnormality Detected" if pred_label == "abnormal" else "No Abnormality Detected"
sub_txt   = (
    "The model has identified potential abnormal features in this scan."
    if pred_label == "abnormal"
    else "The model found no significant abnormal features in this scan."
)

st.markdown(f"""
<div class="result-card {card_cls}">
  <div class="result-title">{icon} {title_txt}</div>
  <div class="result-subtitle">{sub_txt}</div>
  <div style="margin-top:1rem; font-size:0.85rem; color:#8b949e">
      Confidence
      <span style="float:right; font-weight:600; color:#e6edf3">{conf_pct:.1f}%</span>
  </div>
  <div class="conf-bar-wrap">
    <div class="conf-bar-fill {bar_cls}" style="width:{conf_pct:.1f}%"></div>
  </div>
  <div style="font-size:0.78rem; color:#484f58; margin-top:0.3rem">
    Abnormal probability: {abnormal_prob:.4f} &nbsp;|&nbsp; Normal probability: {1-abnormal_prob:.4f}
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# VISUALISATION: Original | Heatmap | Overlay
# ─────────────────────────────────────────────
st.markdown("### Grad-CAM Explainability")
st.caption("Red/yellow regions show where the model focused its attention when making the prediction.")

original, overlaid = overlay_heatmap(pil_image, heatmap)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.patch.set_facecolor("#0d1117")
titles = ["Original Scan", "Attention Heatmap", "Overlay"]
images = [original, heatmap, overlaid]
cmaps  = [None, "jet", None]

for ax, title, img, cmap in zip(axes, titles, images, cmaps):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, color="#8b949e", fontsize=9, pad=8)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout(pad=0.5)
st.image(fig_to_pil(fig), use_column_width=True)
plt.close(fig)

# ─────────────────────────────────────────────
# UPLOAD ANOTHER
# ─────────────────────────────────────────────
st.markdown("""
<div class="info-box">
  <strong>How to interpret:</strong><br>
  • <strong>Confidence ≥ 85%</strong> — High confidence prediction<br>
  • <strong>Confidence 60–85%</strong> — Moderate confidence, consider expert review<br>
  • <strong>Grad-CAM</strong> shows which regions drove the decision (red = highest activation)
</div>

<div class="disclaimer">
  ⚕️ This tool is for <strong>research and educational purposes only</strong>.
  It is <strong>not a medical device</strong> and should not replace professional clinical diagnosis.
</div>
""", unsafe_allow_html=True)
