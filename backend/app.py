"""
Flask inference API for the fetal brain MRI abnormality detector.

Run:
    pip install -r requirements.txt
    python backend_api.py

API:
    GET  /api/health
    GET  /api/model-health
    POST /api/predict  (multipart/form-data with "image")
"""

import base64
import json
import os
import platform
import shutil
import tempfile
import traceback
from pathlib import Path
from threading import Lock, Thread

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

try:
    from .mri_validation import load_mri_reference_profile, validate_mri_like_image
except ImportError:
    from mri_validation import load_mri_reference_profile, validate_mri_like_image


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = (
    BASE_DIR / "results" / "best_model.h5",
    BASE_DIR / "results" / "best_model_phase1.h5",
)
IMG_SIZE = (224, 224)
tf = None
load_model = None
img_to_array = None

# This Flask app is the backend server used by the separate React frontend.
# The frontend sends an uploaded MRI image to this backend, and this backend:
# 1. loads the trained TensorFlow model,
# 2. runs prediction,
# 3. generates Grad-CAM outputs,
# 4. returns JSON for the frontend to display.
app = Flask(__name__)
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
    "https://fetal-brain-abnormalities-checker.netlify.app",
)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})


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
    ensure_ml_dependencies()
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
    ensure_ml_dependencies()
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
    ensure_ml_dependencies()
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
active_model_path = None
model_init_error = None
model_init_lock = Lock()
dependency_init_error = None
model_warmup_started = False
model_warmup_lock = Lock()


def get_model_file_status():
    return [
        {
            "path": str(path),
            "exists": path.exists(),
            "sizeMB": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None,
        }
        for path in MODEL_CANDIDATES
    ]


def ensure_ml_dependencies():
    global tf, load_model, img_to_array, dependency_init_error

    if tf is not None and load_model is not None and img_to_array is not None:
        return

    if dependency_init_error is not None:
        raise RuntimeError(dependency_init_error)

    try:
        import tensorflow as tensorflow_module
        from tensorflow.keras.models import load_model as keras_load_model
        from tensorflow.keras.preprocessing.image import img_to_array as keras_img_to_array

        tf = tensorflow_module
        load_model = keras_load_model
        img_to_array = keras_img_to_array
    except Exception as exc:
        dependency_init_error = f"Failed to import TensorFlow dependencies: {exc}"
        raise RuntimeError(dependency_init_error) from exc


def patch_keras_h5_model_config(model_path):
    """
    Create a temporary H5 copy with Keras 3 InputLayer config adjusted for
    TensorFlow/Keras 2.x loaders. The deployed app only needs inference, so the
    original model file stays untouched.
    """
    import h5py

    temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    shutil.copy2(model_path, temp_path)

    def patch_layer(layer):
        config = layer.get("config", {})
        if layer.get("class_name") == "InputLayer" and "batch_shape" in config:
            config.setdefault("batch_input_shape", config.pop("batch_shape"))

        nested_config = config.get("config")
        if isinstance(nested_config, dict):
            patch_layer(nested_config)

        for key in ("layers", "input_layers", "output_layers"):
            value = config.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        patch_layer(item)

    with h5py.File(temp_path, "r+") as h5_file:
        raw_config = h5_file.attrs.get("model_config")
        if raw_config is None:
            return temp_path

        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")

        model_config = json.loads(raw_config)
        for layer in model_config.get("config", {}).get("layers", []):
            patch_layer(layer)
        h5_file.attrs.modify("model_config", json.dumps(model_config).encode("utf-8"))

    return temp_path


def load_keras_model_for_inference(model_path):
    ensure_ml_dependencies()
    try:
        return load_model(model_path, compile=False)
    except TypeError as exc:
        if "batch_shape" not in str(exc):
            raise

        patched_path = patch_keras_h5_model_config(model_path)
        try:
            return load_model(patched_path, compile=False)
        finally:
            try:
                patched_path.unlink()
            except OSError:
                pass


def ensure_model_ready():
    """
    Lazily load model assets after startup so Gunicorn can bind the port quickly.
    """
    global model, grad_model, active_model_path, model_init_error

    if model is not None and grad_model is not None:
        return True, None

    if model_init_error is not None:
        return False, model_init_error

    with model_init_lock:
        if model is not None and grad_model is not None:
            return True, None

        if model_init_error is not None:
            return False, model_init_error

        chosen_model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
        if chosen_model_path is None:
            searched_paths = ", ".join(str(path) for path in MODEL_CANDIDATES)
            model_init_error = (
                "Model not found. Checked: "
                f"{searched_paths}. Deploy one of these model files with the backend."
            )
            return False, model_init_error

        try:
            ensure_ml_dependencies()
            loaded_model = load_keras_model_for_inference(chosen_model_path)
            loaded_grad_model = build_gradcam(loaded_model)
            load_mri_reference_profile()
            model = loaded_model
            grad_model = loaded_grad_model
            active_model_path = chosen_model_path
            return True, None
        except Exception as exc:
            model_init_error = f"Failed to load model assets: {exc}"
            return False, model_init_error


def start_model_warmup():
    global model_warmup_started

    if model is not None and grad_model is not None:
        return False

    with model_warmup_lock:
        if model_warmup_started:
            return False
        model_warmup_started = True

    def warmup():
        ensure_model_ready()

    Thread(target=warmup, daemon=True).start()
    return True


@app.get("/api/health")
def health():
    # Lightweight health route for Render. Do not load TensorFlow/model here,
    # because Render health checks can fail if model loading takes too long.
    return jsonify(
        {
            "status": "ok",
            "service": "fetal-brain-mri-backend",
            "modelHealth": "/api/model-health",
            "predict": "/api/predict",
        }
    )


@app.get("/api/model-health")
def model_health():
    # Default diagnostic route is lightweight. Add ?load=1 to test TensorFlow
    # and model loading explicitly, because that can be slow/heavy on Render.
    should_load_model = request.args.get("load") == "1"
    ready = model is not None and grad_model is not None
    error_message = model_init_error or dependency_init_error

    if should_load_model:
        start_model_warmup()
        error_message = error_message or "Model warmup started. Refresh this endpoint in 1-3 minutes."

    return jsonify(
        {
            "status": "ok" if ready else "model_error",
            "modelPath": str(active_model_path) if active_model_path is not None else None,
            "modelCandidates": get_model_file_status(),
            "modelLoaded": ready,
            "error": error_message,
            "loadAttempted": should_load_model,
            "warmupStarted": model_warmup_started,
            "pythonVersion": platform.python_version(),
            "project": "Transfer learning-based detection of fetal brain abnormalities in MRI scans",
        }
    ), 200


@app.get("/api/tensorflow-health")
def tensorflow_health():
    try:
        start_model_warmup()
        if tf is None:
            return jsonify(
                {
                    "status": "warming",
                    "message": "TensorFlow/model warmup started. Refresh this endpoint in 1-3 minutes.",
                    "pythonVersion": platform.python_version(),
                }
            )
        return jsonify(
            {
                "status": "ok",
                "tensorflowVersion": tf.__version__,
                "pythonVersion": platform.python_version(),
            }
        )
    except BaseException as exc:
        return jsonify(
            {
                "status": "tensorflow_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "pythonVersion": platform.python_version(),
            }
        ), 500


@app.get("/")
def index():
    return jsonify(
        {
            "status": "ok",
            "service": "fetal-brain-mri-backend",
            "health": "/api/health",
            "modelHealth": "/api/model-health",
            "tensorflowHealth": "/api/tensorflow-health",
            "predict": "/api/predict",
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
    ready, error_message = ensure_model_ready()
    if not ready:
        return jsonify({"error": error_message}), 500

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
