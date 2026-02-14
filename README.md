# Cat vs Dog Image Classification using VGG16 Transfer Learning

This project implements a binary image classification system to distinguish between cats and dogs using the VGG16 convolutional neural network through transfer learning. The model is developed using TensorFlow and Keras and demonstrates the complete deep learning workflow including data preprocessing, model training, evaluation, and prediction.

---

## Project Overview

Image classification is a fundamental task in computer vision. Instead of training a deep neural network from scratch, this project uses transfer learning with a pre-trained VGG16 model, which significantly improves performance and reduces training time.

The model extracts high-level visual features from images and classifies them into two categories: cats and dogs.

---

## Dataset Information

The dataset contains labeled images of cats and dogs organized into training, validation, and testing directories.

Typical dataset structure:

dataset/
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/

Due to GitHub storage limitations, the dataset is not included in this repository.

Dataset download link:

https://www.kaggle.com/datasets/tongpython/cat-and-dog

After downloading, extract the dataset and place it inside the "dataset" folder before running the notebook.

---

## Model Architecture

The classification system uses VGG16 as a feature extractor with a custom classification head.

Key components include:

- Pre-trained VGG16 convolutional base
- Frozen lower layers to retain learned features
- Fully connected dense layers for classification
- Dropout layers to reduce overfitting
- Sigmoid output layer for binary classification

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Features

- Transfer learning using VGG16
- Image preprocessing and normalization
- Training and validation performance tracking
- Visualization of accuracy and loss curves
- Binary classification prediction

---

## Results

The transfer learning approach provides strong classification performance with faster convergence compared to training a CNN from scratch. The model demonstrates effective generalization between training and validation datasets.

---

## Project Structure

```
CatDog_VGG16_Project/
├── Cat vs Dog VGG16.ipynb
├── README.md
└── dataset/
    └── dataset_link.txt
```

---

## How to Run the Project

1. Clone the repository:

git clone https://github.com/yourusername/cat-vs-dog-vgg16.git

2. Install dependencies:

pip install tensorflow numpy matplotlib

3. Download and extract the dataset into the "dataset" folder.

4. Open the notebook and run all cells.

---

## Future Improvements

- Fine-tuning deeper VGG16 layers
- Data augmentation techniques
- Hyperparameter optimization
- Comparison with other architectures such as ResNet or EfficientNet

---

## Author

Ayan Biswas  
B.Tech Computer Science and Engineering  
AI/ML Researcher and Founder of Infiltrix

