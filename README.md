# 🧠 COVID-Xray AI: Deep Learning for COVID-19 Detection

## 📌 Project Overview

This project leverages **Convolutional Neural Networks (CNNs)** to detect **COVID-19 from chest X-ray images**. The aim is to assist early diagnosis using deep learning methods and evaluate multiple architectures like **ResNet18**, **AlexNet**, and **VGG16**.

It also serves as a reference for our academic publication submitted as part of our coursework.

## 🚀 How It Works

### 🧪 1. Dataset

We use public chest X-ray datasets labeled as:
- COVID-19
- Pneumonia
- Normal

### ⚙️ 2. Preprocessing

- Image resizing
- Normalization
- Data augmentation (optional)

### 🧠 3. Models Used

- `ResNet18` – used as the primary model
- `AlexNet` and `VGG16` – for performance comparison
- All models are trained using **PyTorch**

### 📈 4. Evaluation Metrics

- Accuracy
- Precision / Recall
- Confusion Matrix
- ROC Curve
