# React Frontend

This folder contains a separate React frontend for the project:

`Transfer learning-based detection of fetal brain abnormalities in MRI scans`

## Run the backend

From the project root:

```bash
python app.py
```

The API will start on `http://localhost:5000`.

## Run the frontend

From this `frontend` folder:

```bash
npm install
npm run dev
```

The React app will start on `http://localhost:5173`.

## Optional environment variable

If your API runs on a different URL, create a `.env` file in this folder:

```bash
VITE_API_BASE_URL=http://localhost:5000
```

## Deployment

Netlify should build from the repository root using `netlify.toml`. That file points
Netlify to this `frontend` folder and sets:

```bash
VITE_API_BASE_URL=https://fetal-brain-abnormalities.onrender.com
```

Render should use the root `render.yaml`, which installs `backend/requirements.txt`
and starts the Flask API with:

```bash
gunicorn backend.app:app
```

After deploying, open the Render backend URL and check `/api/health`. That route
only checks whether the server is alive. To check TensorFlow and model loading,
open `/api/model-health`. If prediction requests fail from Netlify, make sure
the Netlify site URL is included in the Render `CORS_ORIGINS` environment
variable.
