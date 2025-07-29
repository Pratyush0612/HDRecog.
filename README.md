# 🧠 Handwritten Digit & Character Recognition (EMNIST) – FastAPI

This project implements a complete pipeline for handwritten **digit and character recognition** using machine learning and deep learning models. Built with **FastAPI**, the application exposes a RESTful API that serves a trained CNN model and provides real-time prediction capabilities through a web interface.

---

## 📌 Overview

- Trained multiple ML models (**SVM, RFC, MLP, CNN**) on the **EMNIST dataset** (Extended MNIST: digits + letters).
- Conducted a **comparative study** of traditional ML vs deep learning for OCR tasks.
- Achieved **99% accuracy** using CNN on 5,000+ test samples.
- Deployed via **FastAPI**, integrated with a frontend for real-time image prediction.
- Designed for real-world OCR use cases like **form scanning, test grading, and document digitization**.

---

## 🚀 Features

- 🧠 Model comparison: SVM, RFC, MLP, and CNN
- 📊 Performance evaluation using confusion matrix & accuracy metrics
- 🔄 FastAPI-based prediction API
- 🌐 Frontend interface for uploading handwritten input
- 🧾 Real-time predictions and visual feedback
- 📦 Deployed via [Render](https://render.com)

---

## 🧰 Tech Stack

| Layer        | Tools Used                                     |
|--------------|------------------------------------------------|
| ML Models    | Scikit-learn, TensorFlow, Keras                |
| Backend/API  | FastAPI, Uvicorn                               |
| Data Handling| NumPy, Pandas, Matplotlib                      |
| Frontend     | HTML/CSS, JavaScript *(or add React if used)* |
| Deployment   | Render                                         |

---

## 🧠 Model Comparison

| Model | Accuracy | Training Time | Notes |
|-------|----------|---------------|-------|
| SVM   | ~94%     | Moderate      | Good for digits, weaker on letters |
| RFC   | ~92%     | Fast          | Lightweight, less accurate         |
| MLP   | ~96%     | Moderate      | Decent balance                     |
| CNN   | **99%**  | High          | Best performer on EMNIST           |

---

## 🧪 Use Cases

- Automated form & document digitization
- Bank cheque character recognition
- Exam/test grading systems
- Postal code recognition
- Real-time handwritten input interfaces

---

## 🛠️ API Documentation

### `POST /predict`

**Description:** Predict digit or character from image input

**Request:**
- `Content-Type: multipart/form-data`
- Field: `file` → grayscale image (28x28 recommended)

**Response:**

```json
{
  "prediction": "G"
}
