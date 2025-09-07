# 🧠 Brain Tumor Detection API  

This repository contains a Flask-based application for **classifying brain MRI scans as either `tumor` or `no_tumor`**.  
It combines a trained Convolutional Neural Network (CNN) with a simple API and web interface to make predictions accessible.  

---

## 📂 Project Structure  

Here’s what each key folder and file does:  

- **`notebooks/api/model/brain_mri_model.h5`**  
  Pre-trained CNN model for brain tumor classification. The path can be overridden with the `MODEL_PATH` environment variable.  

- **`app.py`**  
  Main Flask application that handles:  
  - Image upload via web form or API request  
  - Preprocessing with OpenCV + NumPy  
  - Running inference with the trained model  
  - Returning JSON responses with label and confidence score  

- **`static/` & `templates/`**  
  Assets and HTML templates for the simple web interface (upload page + results).  

- **`Dockerfile`**  
  Defines how the app is containerized. Uses Python slim base image, installs dependencies, and runs the app with Gunicorn.  

- **`requirements.api.txt`**  
  Minimal set of dependencies for Docker builds and production.  

- **`requirements.txt`**  
  Full set of dependencies for local development.  

- **`render.yaml`**  
  Configuration for deploying on [Render](https://render.com). Handles environment setup and Docker build instructions.  

---

## 📊 Model Details  

- **Architecture**: Convolutional Neural Network (TensorFlow/Keras)  
- **Classes**: `tumor`, `no_tumor`  
- **Accuracy**: ~94–95% on validation data  
- **Limitations**: Predictions are not 100% accurate and require fine-tuning.  
- **Confidence Scores**: Each prediction includes a probability score for better interpretability.  

---

## 📂 Dataset  

The model was trained on the Kaggle dataset:  
🔗 [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  

---

## 🌐 Features  

- Upload MRI images via browser or API endpoint  
- Binary classification output: **Tumor / No Tumor**  
- Confidence score displayed with each prediction  
- Dockerized for portability and easy deployment  

---

## 💡 Learning Journey  

This project was developed as a **learning experience** to explore:  
- Training deep learning models for medical imaging  
- Serving ML predictions with Flask APIs  
- Building lightweight Docker images for deployment  
- Creating a simple, user-friendly interface for ML outputs  

---

## 📌 Future Improvements  

- Fine-tune the model for higher accuracy  
- Add Grad-CAM visualizations to highlight tumor regions  
- Explore scaling deployment beyond Render  

---
