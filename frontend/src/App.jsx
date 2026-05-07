import { useState } from "react";

// This is the frontend-to-backend connection point.
// Netlify injects VITE_API_BASE_URL at build time. The deployed Render URL is
// kept as a fallback so a missed Netlify env var does not silently call localhost.
const DEFAULT_API_BASE =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:5000"
    : "https://fetal-brain-abnormalities.onrender.com";
const API_BASE = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE).replace(/\/$/, "");

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const waitForModel = async () => {
  const maxAttempts = 12;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    const response = await fetch(`${API_BASE}/api/model-health?load=1`, {
      method: "GET",
      cache: "no-store",
    });

    const data = await response.json();
    if (data.modelLoaded) {
      return;
    }

    if (data.error && !String(data.error).toLowerCase().includes("warmup")) {
      throw new Error(data.error);
    }

    await sleep(10000);
  }

  throw new Error("The backend model is still warming up. Please try again in a minute.");
};

// These cards are just UI content that explain the project idea on the page.
const featureCards = [
  {
    title: "Transfer Learning Pipeline",
    text: "Uses a fine-tuned VGG16 backbone to separate normal fetal MRI scans from abnormal ones.",
  },
  {
    title: "Explainable Inference",
    text: "Generates Grad-CAM attention maps so users can inspect the regions influencing a prediction.",
  },
  {
    title: "Binary Clinical Triage View",
    text: "Frames the task as normal vs abnormal to match the current dataset quality and project scope.",
  },
];

function App() {
  // selectedFile: the MRI image chosen by the user
  // previewUrl: temporary browser URL used to preview the chosen image
  // result: JSON response returned by the backend after prediction
  // loading: controls button text / waiting state
  // error: shows backend or validation errors in the UI
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // This controls result panel styling:
  // - "alert" for abnormal predictions
  // - "safe" for normal predictions
  // - "neutral" before any result arrives
  const predictionTone = !result ? "neutral" : result.prediction === "abnormal" ? "alert" : "safe";

  const handleFileChange = (event) => {
    // Triggered when the user picks a new image in the upload box.
    // It resets any old prediction and creates a browser preview.
    const file = event.target.files?.[0];
    setResult(null);
    setError("");

    if (!file) {
      setSelectedFile(null);
      setPreviewUrl("");
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleSubmit = async (event) => {
    // Triggered when the user clicks "Run Prediction".
    // This is the exact place where frontend connects to backend.
    event.preventDefault();

    if (!selectedFile) {
      setError("Please choose an MRI image first.");
      return;
    }

    const formData = new FormData();
    // "image" must match the Flask backend key in request.files["image"].
    formData.append("image", selectedFile);

    try {
      setLoading(true);
      setError("");
      setResult(null);

      setError("Starting backend model. This can take 1-3 minutes after Render wakes up.");
      await waitForModel();
      setError("");

      // Frontend -> backend API request:
      // Sends the uploaded MRI image to Flask at /api/predict.
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: formData,
      });

      const contentType = response.headers.get("content-type") || "";
      const rawBody = await response.text();
      let data = {};

      // The backend normally returns JSON containing label, confidence,
      // probabilities, and Grad-CAM images, but local setup errors can return
      // HTML or an empty body instead.
      if (rawBody) {
        if (contentType.includes("application/json")) {
          try {
            data = JSON.parse(rawBody);
          } catch {
            throw new Error("Backend returned invalid JSON. Check the local backend terminal for the real error.");
          }
        } else {
          throw new Error(
            "Backend did not return JSON. Make sure the Flask API is running on "
            + `${API_BASE} and check the backend terminal for errors.`
          );
        }
      }

      if (!response.ok) {
        throw new Error(data.error || "Prediction request failed.");
      }

      // Store backend response so the UI can render the result panel.
      setResult(data);
    } catch (err) {
      // Any backend/network error is shown to the user in the red message box.
      const message = err.message || "Something went wrong while connecting to the backend.";
      setError(
        message === "Failed to fetch"
          ? `Could not reach the backend at ${API_BASE}. Check that the Render service is running and CORS allows this Netlify site.`
          : message
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-shell">
      {/* Decorative background shapes only */}
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <header className="hero">
        {/* Project introduction area */}
        <div className="hero-copy">
          <span className="eyebrow">Research Project Dashboard</span>
          <h1>Transfer learning-based detection of fetal brain abnormalities in MRI scans</h1>
          <p>
            A responsive React interface for your fetal brain MRI project, connected to a Python
            backend that serves VGG16 predictions and Grad-CAM visual explanations.
          </p>
        </div>
        <div className="hero-panel">
          {/* Quick technology summary */}
          <div className="hero-stat">
            <span>Model task</span>
            <strong>Normal vs Abnormal</strong>
          </div>
          <div className="hero-stat">
            <span>Backend</span>
            <strong>Flask + TensorFlow</strong>
          </div>
          <div className="hero-stat">
            <span>Frontend</span>
            <strong>React + Vite</strong>
          </div>
        </div>
      </header>

      <section className="feature-grid">
        {/* Static cards explaining what the project does */}
        {featureCards.map((card) => (
          <article className="feature-card" key={card.title}>
            <h2>{card.title}</h2>
            <p>{card.text}</p>
          </article>
        ))}
      </section>

      <main className="workspace-grid">
        <section className="panel upload-panel">
          {/* Upload section: user chooses MRI image and submits it to backend */}
          <div className="panel-header">
            <span className="section-tag">Upload</span>
            <h2>Analyze an MRI scan</h2>
            <p>Upload a JPG, JPEG, PNG, or BMP image and send it to the inference API.</p>
          </div>

          <form onSubmit={handleSubmit} className="upload-form">
            <label className="dropzone">
              {/* File input is hidden visually by CSS, but clicking the dropzone opens it */}
              <input
                type="file"
                accept=".jpg,.jpeg,.png,.bmp"
                onChange={handleFileChange}
              />
              <div>
                <strong>{selectedFile ? selectedFile.name : "Choose fetal MRI image"}</strong>
                <span>Supports grayscale and RGB scans. Optimized for 224 x 224 inference.</span>
              </div>
            </label>

            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Analyzing scan..." : "Run Prediction"}
            </button>
          </form>

          {/* Error box shown when:
              - user submits with no file
              - backend returns an error
              - network/API call fails
          */}
          {error ? <div className="message error-message">{error}</div> : null}

          {previewUrl ? (
            <div className="preview-card">
              {/* Local browser preview before backend prediction */}
              <div className="preview-header">
                <span className="section-tag">Preview</span>
                <p>Selected image</p>
              </div>
              <img src={previewUrl} alt="Uploaded MRI preview" />
            </div>
          ) : null}
        </section>

        <section className={`panel result-panel tone-${predictionTone}`}>
          {/* Result section:
              waits for backend response, then renders prediction + images
          */}
          <div className="panel-header">
            <span className="section-tag">Inference</span>
            <h2>Prediction summary</h2>
            <p>The prediction card updates after the backend processes the uploaded scan.</p>
          </div>

          {!result ? (
            <div className="empty-state">
              {/* Placeholder before the first prediction */}
              <p>No prediction yet.</p>
              <span>Upload a scan to see abnormal probability, normal probability, and Grad-CAM output.</span>
            </div>
          ) : (
            <>
              {/* Main prediction summary using backend JSON values */}
              <div className="result-banner">
                <div>
                  <span className="result-label">
                    {result.prediction === "abnormal" ? "Potential abnormality detected" : "No major abnormality detected"}
                  </span>
                  <h3>{Math.round(result.confidence * 100)}% confidence</h3>
                </div>
                <div className="probability-stack">
                  <p>Abnormal: {(result.abnormalProbability * 100).toFixed(1)}%</p>
                  <p>Normal: {(result.normalProbability * 100).toFixed(1)}%</p>
                </div>
              </div>

              <p className="summary-text">{result.summary}</p>

              <div className="visual-grid">
                {/* These three images come from backend_api.py:
                    result.images.original
                    result.images.heatmap
                    result.images.overlay
                */}
                <figure className="visual-card">
                  <img src={result.images.original} alt="Original MRI" />
                  <figcaption>Original scan</figcaption>
                </figure>
                <figure className="visual-card">
                  <img src={result.images.heatmap} alt="Grad-CAM heatmap" />
                  <figcaption>Attention heatmap</figcaption>
                </figure>
                <figure className="visual-card">
                  <img src={result.images.overlay} alt="Grad-CAM overlay" />
                  <figcaption>Overlay view</figcaption>
                </figure>
              </div>
            </>
          )}
        </section>
      </main>

      <section className="panel info-strip">
        {/* Extra explanation panel for the project purpose */}
        <div>
          <span className="section-tag">Use Note</span>
          <h2>How this frontend supports your project idea</h2>
        </div>
        <p>
          This UI presents the project as a transfer learning-based MRI screening assistant,
          showing prediction confidence and explainability side by side in a mobile-friendly layout.
        </p>
      </section>

      <footer className="footer-note">
        {/* Final disclaimer shown at the bottom of the page */}
        For research and educational use only. This interface does not replace clinical diagnosis.
      </footer>
    </div>
  );
}

export default App;
