# 🌿 Leaf Disease Classification using Deep Learning

Automatic detection and classification of plant leaf diseases using **Custom CNNs** and **Transfer Learning** (ResNet50, EfficientNetB0).  
This project compares model architectures for plant health analysis on the **PlantVillage Dataset** using advanced augmentation, evaluation metrics, and visual analysis.

---

## 🧩 Table of Contents

1. [Overview](#overview)  
2. [Motivation](#motivation)  
3. [Dataset](#dataset)  
4. [Preprocessing](#preprocessing)  
5. [Model Architectures](#model-architectures)  
6. [Training Details](#training-details)  
7. [Results](#results)  
8. [Visualizations](#visualizations)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Reproducibility](#reproducibility)  
12. [Project Structure](#project-structure)  
13. [Future Work](#future-work)  
14. [References](#references)  
15. [License](#license)  
16. [Author](#author)

1. Overview
2. Motivation
3. Dataset
4. Preprocessing
5. Model Architectures
6. Training Details
7. Results
8. Visualizations
9. Installation ✅
10. Usage ✅
11. Reproducibility
12. Project Structure
13. Future Work
14. References
15. License
16. Author

---

## 📖 Overview

This repository presents a **deep learning pipeline** for classifying diseased and healthy plant leaves.  
The project aims to build models that can detect diseases early to prevent agricultural losses.

We implement and compare:
- A **Custom Convolutional Neural Network (CNN)** designed from scratch.  
- **Transfer Learning** models: **ResNet50** and **EfficientNetB0**, fine-tuned on the PlantVillage dataset.

---

## 🌱 Motivation

Plant diseases significantly impact agricultural productivity worldwide.  
Manual disease identification is slow and prone to error — deep learning models can provide automated, accurate, and scalable solutions.  

This project demonstrates:
- How to apply CNNs and transfer learning to image classification problems.  
- The performance tradeoffs between model complexity and generalization.  
- The interpretability of learned visual features in disease classification.

---

## 📂 Dataset

- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Classes:** 10 leaf categories (9 diseased + 1 healthy)  
- **Image Resolution:** 224×224 pixels  
- **Format:** RGB  
- **Split:** 80% train / 10% validation / 10% test  

**Class Distribution:**

![Dataset Distribution](images/dataset.png)

---

## 🧼 Preprocessing

- **Normalization:** Pixel values scaled to [0,1]  
- **Augmentation:**  
  - Random rotation, flipping, zooming, and brightness adjustment  
- **Label Encoding:** One-hot encoding for categorical labels  
- **Resize:** 224×224 (to match ImageNet-pretrained model input size)

---

## 🧠 Model Architectures

### 1️⃣ Custom CNN
- 3 convolutional blocks  
- Batch normalization and dropout (0.3–0.5)  
- Fully connected dense layers  
- Softmax output layer  

### 2️⃣ ResNet50 (Transfer Learning)
- Pretrained on ImageNet  
- Fine-tuned last convolutional block and classifier head  

### 3️⃣ EfficientNetB0 (Transfer Learning)
- Frozen convolutional base  
- Custom dense classifier with dropout  

| Model | Trainable Params | Pretrained | Regularization |
|--------|------------------|-------------|----------------|
| CNN | 1.2M | No | Dropout + BN |
| ResNet50 | 23M | Yes | L2 + Dropout |
| EfficientNetB0 | 5.3M | Yes | Dropout |

---

## ⚙️ Training Details

| Setting | Value |
|----------|-------|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Learning Rate | 1e-4 (decay schedule) |
| Batch Size | 32 |
| Epochs | 30 |
| Metrics | Accuracy, F1-score, Precision, Recall |

**Callbacks:**
- EarlyStopping (patience=5)  
- ReduceLROnPlateau (factor=0.2)  
- ModelCheckpoint (best validation accuracy)

---

## 📊 Results

| Model | Val Accuracy | Test Accuracy | F1-Score | Notes |
|--------|---------------|----------------|----------|--------|
| Custom CNN | 0.91 | 0.90 | 0.89 | Decent baseline |
| ResNet50 | 0.96 | 0.95 | 0.94 | Great generalization |
| EfficientNetB0 | **0.97** | **0.96** | **0.96** | Best performance |

**Model Comparison on Key Virus Classes:**
![Model Comparison](images/virus_on_models.png)

---

## 📈 Visualizations

**Training Curves:**
![Training Curves](images/training_curves.png)

**Confusion Matrix:**
![Confusion Matrix](images/confusion_matrix.png)

**Prediction Examples:**
![Predictions](images/predictions.png)

---


## 🧪 Reproducibility

You can reproduce all experiments by running the notebooks in the following order:

1. `1_data_preprocessing.ipynb` – prepares and augments the dataset  
2. `2_model_training.ipynb` – trains models (Custom CNN, ResNet, EfficientNet)  
3. `3_model_evaluation.ipynb` – evaluates model performance and generates plots  

All visualizations are saved in the `images/` folder.

---

## ⚙️ Installation

This project was developed in **Python 3.11** and requires the following main libraries:

- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

To install dependencies, run:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## 🚀 Usage

After installation, launch Jupyter Notebook:

```bash
jupyter notebook
```

