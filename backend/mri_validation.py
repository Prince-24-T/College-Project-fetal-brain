import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "dataset"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
PROFILE_FEATURE_KEYS = (
    "channel_delta",
    "mean_saturation",
    "edge_density",
    "largest_area_ratio",
    "center_offset_ratio",
    "bbox_fill_ratio",
    "small_component_count",
    "border_dark_ratio",
    "gray_mean",
    "gray_std",
)


def iter_dataset_image_paths(dataset_root=DATASET_ROOT):
    for split in ("train", "valid", "test"):
        split_dir = Path(dataset_root) / split
        if not split_dir.is_dir():
            continue
        for base, _, files in os.walk(str(split_dir)):
            for filename in files:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    yield os.path.join(base, filename)


@lru_cache(maxsize=1)
def load_mri_reference_profile(dataset_root=DATASET_ROOT):
    feature_rows = []
    histograms = []

    for image_path in iter_dataset_image_paths(dataset_root):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            features = extract_mri_features_from_rgb(rgb)
            feature_rows.append([features[key] for key in PROFILE_FEATURE_KEYS])
            histograms.append(features["histogram"])
        except Exception:
            continue

    if not feature_rows:
        return None

    values = np.array(feature_rows, dtype=np.float32)
    histograms = np.array(histograms, dtype=np.float32)

    return {
        "low": dict(zip(PROFILE_FEATURE_KEYS, np.percentile(values, 5, axis=0))),
        "high": dict(zip(PROFILE_FEATURE_KEYS, np.percentile(values, 95, axis=0))),
        "median": dict(zip(PROFILE_FEATURE_KEYS, np.percentile(values, 50, axis=0))),
        "hist_mean": np.mean(histograms, axis=0),
        "hist_l1_p95": float(np.percentile(np.sum(np.abs(histograms - np.mean(histograms, axis=0)), axis=1), 95)),
    }


def extract_mri_features_from_rgb(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    channel_delta = float(
        (
            np.mean(np.abs(rgb[:, :, 0].astype(np.float32) - rgb[:, :, 1].astype(np.float32)))
            + np.mean(np.abs(rgb[:, :, 1].astype(np.float32) - rgb[:, :, 2].astype(np.float32)))
            + np.mean(np.abs(rgb[:, :, 0].astype(np.float32) - rgb[:, :, 2].astype(np.float32)))
        )
        / 3.0
    )
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary == 255) > 0.5:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest_area_ratio = 0.0
    center_offset_ratio = 1.0
    bbox_fill_ratio = 0.0
    small_component_count = 0

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1
        largest_area = float(stats[largest_idx, cv2.CC_STAT_AREA])
        image_area = float(gray.shape[0] * gray.shape[1])
        largest_area_ratio = largest_area / image_area

        cx, cy = centroids[largest_idx]
        nx = float((cx / gray.shape[1]) - 0.5)
        ny = float((cy / gray.shape[0]) - 0.5)
        center_offset_ratio = float(np.sqrt(nx * nx + ny * ny))

        width = float(stats[largest_idx, cv2.CC_STAT_WIDTH])
        height = float(stats[largest_idx, cv2.CC_STAT_HEIGHT])
        bbox_fill_ratio = largest_area / max(width * height, 1.0)

        small_component_count = int(np.sum(areas < image_area * 0.01))

    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    border_dark_ratio = float(np.mean(border_pixels < 35))
    histogram = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten().astype(np.float32)
    histogram /= float(np.sum(histogram) + 1e-8)

    return {
        "channel_delta": channel_delta,
        "mean_saturation": mean_saturation,
        "edge_density": edge_density,
        "largest_area_ratio": largest_area_ratio,
        "center_offset_ratio": center_offset_ratio,
        "bbox_fill_ratio": bbox_fill_ratio,
        "small_component_count": small_component_count,
        "border_dark_ratio": border_dark_ratio,
        "gray_mean": float(np.mean(gray)),
        "gray_std": float(np.std(gray)),
        "histogram": histogram,
    }


def extract_mri_features(pil_image):
    return extract_mri_features_from_rgb(np.array(pil_image.convert("RGB")))


def validate_mri_like_image(pil_image):
    """
    Reject unrelated images before model inference.
    Returns (is_valid, reason).
    """
    features = extract_mri_features(pil_image)
    profile = load_mri_reference_profile()

    severe_failures = []
    failures = []
    accept_signals = 0

    if features["channel_delta"] <= 10 and features["mean_saturation"] <= 28:
        accept_signals += 1
    else:
        severe_failures.append("high color content")

    if 0.15 <= features["border_dark_ratio"] <= 0.98:
        accept_signals += 1
    else:
        failures.append("background is not MRI-like")

    if 0.02 <= features["largest_area_ratio"] <= 0.95:
        accept_signals += 1
    else:
        failures.append("invalid foreground shape")

    if features["center_offset_ratio"] <= 0.33:
        accept_signals += 1
    else:
        failures.append("off-center main structure")

    if 0.12 <= features["bbox_fill_ratio"] <= 0.98:
        accept_signals += 1
    else:
        failures.append("foreground region is not scan-like")

    if features["edge_density"] <= 0.22:
        accept_signals += 1
    else:
        failures.append("natural-image edge pattern")

    if profile is not None:
        outlier_count = 0
        outlier_labels = []
        for key in PROFILE_FEATURE_KEYS:
            value = features[key]
            low = float(profile["low"][key])
            high = float(profile["high"][key])
            padding = max((high - low) * 0.15, 1e-6)
            if value < low - padding or value > high + padding:
                outlier_count += 1
                outlier_labels.append(key)

        hist_distance = float(np.sum(np.abs(features["histogram"] - profile["hist_mean"])))
        if hist_distance > max(profile["hist_l1_p95"] * 1.1, 0.35):
            outlier_count += 1
            outlier_labels.append("intensity histogram")

        if outlier_count <= 4:
            accept_signals += 1
        elif outlier_count >= 7:
            label_map = {
                "edge_density": "natural-image edge pattern",
                "largest_area_ratio": "invalid foreground shape",
                "center_offset_ratio": "off-center main structure",
                "bbox_fill_ratio": "foreground region is not scan-like",
                "small_component_count": "too many disconnected regions",
                "border_dark_ratio": "background is not MRI-like",
                "gray_mean": "brightness pattern is not MRI-like",
                "gray_std": "contrast pattern is not MRI-like",
                "channel_delta": "high color content",
                "mean_saturation": "high color content",
                "intensity histogram": "intensity distribution is not MRI-like",
            }
            for key in outlier_labels:
                failures.append(label_map.get(key, "image pattern is outside MRI dataset"))
    else:
        if features["small_component_count"] > 20:
            failures.append("too many disconnected regions")

    if features["small_component_count"] <= 20:
        accept_signals += 1

    strong_mri_like = (
        features["channel_delta"] <= 10
        and features["mean_saturation"] <= 28
        and features["edge_density"] <= 0.22
        and features["center_offset_ratio"] <= 0.33
        and features["largest_area_ratio"] >= 0.02
    )

    if strong_mri_like and accept_signals >= 5:
        return True, ""

    should_reject = bool(severe_failures) or len(failures) >= 3 or accept_signals <= 3
    if should_reject:
        reasons = []
        for reason in severe_failures + failures:
            if reason not in reasons:
                reasons.append(reason)
        reasons = reasons[:2]
        return (
            False,
            "Uploaded image does not appear to be a fetal brain MRI slice "
            f"({', '.join(reasons)}).",
        )

    return True, ""
