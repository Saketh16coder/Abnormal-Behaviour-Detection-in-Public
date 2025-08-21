# Abnormal-Behaviour-Detection-in-Public



A deep learning-based real-time violence detection system using MobileNetV2 and ResNet50 on surveillance video frames.

## 🧠 Project Overview

This project classifies video frames into **violent** or **non-violent** categories using transfer learning with state-of-the-art convolutional neural networks. It is aimed at enhancing public safety by detecting real-life violent activities in public spaces through surveillance footage.

## 📂 Dataset

- **Categories:** Violence / Non-Violence
- **Image Format:** Extracted frames from video clips
- **Frame Sampling:** Every 10th frame
- **Image Size:** 64×64 pixels
- **Train/Test Split:** Handled using `train_test_split` or `ImageDataGenerator`

## 🏗️ Models Used

### ✅ MobileNetV2 (Pretrained)
- Fine-tuned on our dataset
- Efficient and lightweight, suitable for real-time deployment

### ✅ ResNet50 (Pretrained)
- Deeper architecture for capturing complex features
- Fine-tuned with a custom classification head

Both models use:
- `GlobalAveragePooling2D`
- `Dense` output layer with sigmoid activation for binary classification

## ⚙️ Training Details

- **Framework:** TensorFlow / Keras
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy
- **Callbacks:** EarlyStopping, ModelCheckpoint
- **Visualization:** Confusion Matrix, Accuracy/Loss Plots

## 📈 Results

- High training and validation accuracy (>90%)
- Balanced performance between violent and non-violent categories
- Confusion matrix and classification report used for evaluation

## 📊  Output


Model: MobileNetV2
Accuracy: 94.8%
Precision: 93.5%
Recall: 92.7%
