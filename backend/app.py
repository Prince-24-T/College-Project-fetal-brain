"""
Flask inference API for the fetal brain MRI abnormality detector.

Run:
    pip install -r requirements.txt
    python backend_api.py

API:
    GET  /api/health
    POST /api/predict  (multipart/form-data with "image")
"""

import base64
import os

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from mri_validation import load_mri_reference_profile, validate_mri_like_image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


MODEL_PATH = "results/best_model.h5"
IMG_SIZE = (224, 224)

# This Flask app is the backend server used by the separate React frontend.
# The frontend sends an uploaded MRI image to this backend, and this backend:
# 1. loads the trained TensorFlow model,
# 2. runs prediction,
# 3. generates Grad-CAM outputs,
# 4. returns JSON for the frontend to display.
app = Flask(__name__)
CORS(app)


def image_to_data_url(image_array):
    """Encode a uint8 RGB or grayscale numpy image as a PNG data URL."""
    # The React frontend expects images it can render immediately in the browser.
    # Returning a base64 data URL lets the frontend show original / heatmap / overlay
    # without saving temporary image files on disk.
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 3 and image_array.shape[2] == 3:
        success, buffer = cv2.imencode(".png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    else:
        success, buffer = cv2.imencode(".png", image_array)

    if not success:
        raise ValueError("Failed to encode image output.")

    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def prepare_rgb_image(uploaded_file):
    # This function converts the uploaded MRI image into the exact tensor format
    # expected by the model:
    # - PIL image for display work
    # - resized to 224x224
    # - converted to array
    # - normalized to [0, 1]
    # - batch dimension added so TensorFlow can predict on it
    pil_image = Image.open(uploaded_file).convert("RGB")
    resized = pil_image.resize(IMG_SIZE)
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return pil_image, img_array


def overlay_heatmap(pil_image, heatmap, alpha=0.45):
    # This creates the three visual outputs shown in the React frontend:
    # 1. original resized image
    # 2. colored heatmap
    # 3. overlay image that blends the scan with the heatmap
    image = np.array(pil_image.resize(IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return image, heatmap_uint8, overlay


def build_gradcam(model):
    # Grad-CAM needs access to:
    # - the last convolution layer output
    # - the final model output
    # This helper creates a smaller model that returns both in one forward pass.
    last_conv = model.get_layer("block5_conv3")
    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output],
    )


def compute_gradcam(grad_model, img_array):
    """
    Returns a Grad-CAM heatmap targeted to the predicted class for sigmoid output.
    """
    # This is the main explainability function.
    # It computes:
    # - abnormal probability from the sigmoid output
    # - which class the model predicted
    # - gradients of that chosen class with respect to the last conv layer
    # - a normalized Grad-CAM heatmap
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        abnormal_prob = predictions[:, 0]
        predicted_abnormal = abnormal_prob[0] > 0.5
        class_channel = abnormal_prob if predicted_abnormal else (1 - abnormal_prob)

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_outputs[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(abnormal_prob[0])


model = None
grad_model = None

# When the backend starts, it loads the trained model one time.
# That way the frontend can make many requests without reloading the model for every image.
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    grad_model = build_gradcam(model)
    load_mri_reference_profile()


@app.get("/api/health")
def health():
    # Frontend or developer can call this route to check whether:
    # - the backend server is running
    # - the model file exists
    # - the model loaded successfully
    return jsonify(
        {
            "status": "ok" if model is not None else "model_missing",
            "modelPath": MODEL_PATH,
            "modelLoaded": model is not None,
            "project": "Transfer learning-based detection of fetal brain abnormalities in MRI scans",
        }
    )


@app.post("/api/predict")
def predict():
    # This is the main backend route used by the React frontend.
    #
    # Frontend connection:
    # In frontend/src/App.jsx, the submit handler sends the selected file here:
    #     fetch(`${API_BASE}/api/predict`, { method: "POST", body: formData })
    #
    # The form field name is "image", so the backend reads request.files["image"].
    if model is None or grad_model is None:
        return jsonify({"error": f"Model not found at {MODEL_PATH}. Run step1_train_model.py first."}), 500

    if "image" not in request.files:
        return jsonify({"error": 'No image uploaded. Send multipart/form-data with field name "image".'}), 400

    uploaded = request.files["image"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename received."}), 400

    try:
        # Step 1: preprocess the uploaded image for model inference
        pil_image, img_array = prepare_rgb_image(uploaded)
        is_mri_like, rejection_reason = validate_mri_like_image(pil_image)
        if not is_mri_like:
            return jsonify({"error": rejection_reason + " Please upload a valid fetal brain MRI image."}), 422

        # Step 2: get Grad-CAM heatmap and abnormal probability from the model
        heatmap, abnormal_prob = compute_gradcam(grad_model, img_array)

        # Step 3: convert model probability into a human-readable label and confidence
        pred_label = "abnormal" if abnormal_prob > 0.5 else "normal"
        normal_prob = 1 - abnormal_prob
        confidence = abnormal_prob if pred_label == "abnormal" else normal_prob

        # Step 4: prepare visual outputs for the frontend
        original, heatmap_uint8, overlay = overlay_heatmap(pil_image, heatmap)

        # Step 5: return JSON that the React frontend can render.
        # In App.jsx these values are used in:
        # - result.prediction
        # - result.confidence
        # - result.abnormalProbability
        # - result.normalProbability
        # - result.images.original / heatmap / overlay
        response = {
            "prediction": pred_label,
            "confidence": round(float(confidence), 4),
            "abnormalProbability": round(float(abnormal_prob), 4),
            "normalProbability": round(float(normal_prob), 4),
            "summary": (
                "The model has identified potential abnormal features in this scan."
                if pred_label == "abnormal"
                else "The model found no major abnormal features in this scan."
            ),
            "images": {
                "original": image_to_data_url(original),
                "heatmap": image_to_data_url(heatmap_uint8),
                "overlay": image_to_data_url(overlay),
            },
        }
        return jsonify(response)
    except Exception as exc:
        # Any backend error is sent back as JSON so the frontend can show it
        # inside its error message box instead of failing silently.
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # Run the backend API locally on port 5000.
    # The React frontend is written to call this URL by default:
    #     http://localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
