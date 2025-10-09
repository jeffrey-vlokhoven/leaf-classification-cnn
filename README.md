# 🌿 Leaf Disease Classification using CNN

Identify and classify the health condition of plant leaves using **Custom CNN** and Transfer Learning** (ResNet50, EfficientNetB0).
This project compares the efficiency of model architectures on the **Planet Village Dataset** using augmentation, evaluation metrics, and visual analysis. 

---

## Table of Contents
1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Dataset](#dataset)
4. [Model Architectures](#model-architectures)
5. [Results](#results)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Future Work](#future-work)
9. [Project Structure](#project-structure)
10. [License](#license)


---

## Overview

This repository presents a **deep learning pipeline** for classifying healthy and diseased plant leaves. 
The project aims to optimize disease detection to prevent agricultural losses and improve quality of life. 

We implement and compare: 
- A **Custom CNN** designed from scratch.
- **Transfer Learning models:** ResNet50 and EfficientNetB0

---

## Motivation

Plant diseases have a crucial impact on agricultural productivity worldwide. Manual disease identification is low and prone to human error, and the solution are deep learning models which automate the process and make accurate predictions.

This project demonstrates:
- How to apply CNNs and transfer learning to image classification problems.
- How different model architectures conclude different results.
- How to choose the optimal model.

---

## Dataset

- **Source:** [Plant Village Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- **Size:** 16,510 images
- **Classes:** 15 leaf classes (2 pepper, 3 potato, and 10 tomato)  
- **Image Resolution:** 256×256 pixels
- **Format:** RGB  
- **Split:** 80% train / 10% validation / 10% test

**Class Distribution:**

![Dataset Distribution](images/dataset.png)

---

## Model Architectures

### 1️⃣ Custom CNN

**Normalization:** Pixel values scaled to [0,1]  
**Augmentation:** Random flipping, rotation, and contrast adjustment

This model uses 4 convolutional blocks, each consisting of:
- Conv2D
- BatchNormalization
- ReLU activation
- MaxPooling2D
Example:
```python
x = Conv2D(64, (3, 3),padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)
```
After that, apply a global average pooling layer, followed by dense layers with L2 regularization and dropouts. 
```python
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu',kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu',kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.5)(x)
```
Concluding with a softmax output layer.
```python
outputs = Dense(len(class_names), activation='softmax')(x)
```
**Resize:** 224×224 (to match ImageNet-pretrained model input size)

### 2️⃣ Transfer Learning
ResNet50 and EfficientNetB0 have the same ending design structure. 
In general, 
- **Resize:** 224×224 (to match model input size)
- **Pretrained Weights:** ImageNet
- **Frozen Base:** Only the classifier head is trained
- **Classifier Head:** GlobalAveragePooling → Dense layers → Dropout → Softmax

Example of ResNet:
```python
base_model_resnet.trainable = False   # freeze pretrained weights

model_resnet = models.Sequential([
    base_model_resnet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax') 
])
```
### Model Architectures comparision
<table>
  <thead>
    <tr><th>Model</th><th>Trainable Params</th><th>Pretrained</th><th>Regularization</th></tr>
  </thead>
  <tbody>
    <tr><td>CNN</td><td>0.5M</td><td>No</td><td>Dropout + BN + L2</td></tr>
    <tr><td>ResNet50</td><td>23.9M</td><td>Yes</td><td>Dropout</td></tr>
    <tr><td>EfficientNetB0</td><td>4.0M</td><td>Yes</td><td>Dropout</td></tr>
  </tbody>
</table>

---

## Results
The following section shows my test results of my three models on two test sets.

**Original test set:**
<table>
    <thead>
      <tr><th>Model</th><th>Accuracy</th><th>Loss</th></tr>
    </thead>
    <tbody>
      <tr><td>own CNN</td><td>0.898</td><td>0.407</td></tr>
      <tr><td>ResNet50</td><td>0.967</td><td>0.120</td></tr>
      <tr><td>EfficientNetB0</td><td>0.951</td><td>0.156</td></tr>
    </tbody>
</table>

Model Comparison on Dangerous Virus Classes:

<p align="center">
  <img src="images/virus_on_models.png" alt="virus pie chart" style="height: auto; width: auto;" />
</p>

**Augmented test set: (user-oriented)**
<table>
    <thead>
      <tr><th>Model</th><th>Accuracy</th><th>Loss</th></tr>
    </thead>
    <tbody>
      <tr><td>own CNN</td><td>0.459</td><td>3.496</td></tr>
      <tr><td>ResNet50</td><td>0.556</td><td>2.814</td></tr>
      <tr><td>EfficientNetB0</td><td>0.708</td><td>1.003</td></tr>
    </tbody>
</table>

Model Comparison on Dangerous Virus Classes:

<p align="center">
  <img src="images/virus_on_models_aug.png" alt="virus pie chart" style="height: auto; width: auto;" />
</p>

For a detailed analysis and evaluation of this project, see the [Analysis](analysis.md) page.

---

## Installation

This project was developed in **Python 3.13.5** and requires the following main libraries:
- **TensorFlow:** 2.20.0
- **NumPy:** 2.0.1
- **Matplotlib:** 3.10.6
- **scikit-learn:** 1.7.1

To install dependencies, run:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```
After installation, launch Jupyter Notebook:

```bash
jupyter notebook
```

## Usage

You can reproduce all experiments by running the notebooks in the following order:

1. `notebook/PlanetVillage.ipynb` – prepares and augments the dataset, and trains my own constructed CNN  
2. `notebook/transferLearning.ipynb` – trains transfer learning models (ResNet, EfficientNet)  
3. `notebook/comparison.ipynb` – evaluates model performance and generates plots

All visualizations are saved in the `images/` folder.

## Project Structure

Below is an overview of the repository layout:

```
leaf-classification-cnn/
├── data/
│ └──archive.zip
├── images/
├── model/
│ └──cnn_mymodel_extendedFinal2.0.keras
│ └──efficient.keras
│ └──resnet.keras
├── notebooks/
│ └──PlanetVillage.ipynb
│ └──comparison.ipynb
│ └──transferLearning.ipynb
├── ANALYSIS.md
├── LICENSE
├── README.md
```

## Future Work

We can explore the trade-off between high-quality (original) images and user-quality (augmented) images, as depending on the data set we select different models. 

For detailed reasoning and motivation, see the last section of [Analysis](analysis.md).

## License

This project is licensed under the [MIT License](LICENSE).

