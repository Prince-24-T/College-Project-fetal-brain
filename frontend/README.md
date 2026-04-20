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
