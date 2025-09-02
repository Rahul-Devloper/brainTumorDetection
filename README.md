# Brain Tumor Detection API

This repository contains a small Flask application for classifying brain MRI images as tumor or no_tumor.

## Deploy on Render

1. Add your trained `brain_mri_model.h5` under `notebooks/api/model/` or set the `MODEL_PATH` environment variable in Render.
2. Push the repository to a Git provider (GitHub, GitLab, etc.).
3. From the Render dashboard, create a **Web Service** and select the repository.
4. Render will use the provided `Dockerfile` and `render.yaml` to build and run the service. No port configuration is needed; the app binds to `$PORT` automatically.
5. Once deployed, open the service URL in a browser to access a simple upload form, or send POST requests to `/predict` with an image file.

## Local development

Build and run the container locally:

Build and run the container locally:

```bash
docker build -t brain-tumor-app .
docker run -p 5000:5000 brain-tumor-app
docker run -e PORT=5000 -p 5000:5000 brain-tumor-app
```

Then visit `http://localhost:5000` in your browser.