# Fashion MNIST Image Classification

## Overview
This repository contains implementations of image classification on the Fashion MNIST dataset using Convolutional Neural Networks (CNNs) in both PyTorch and TensorFlow. The Fashion MNIST dataset comprises 70,000 images of fashion items categorized into 10 classes.

## Contents

### FashionMNIST_PyTorch.ipynb
- This notebook implements image classification using a CNN in PyTorch.
- **Key Steps**:
  1. **Import Libraries**: Necessary libraries such as PyTorch, NumPy, Matplotlib, and torchvision are imported.
  2. **Define Model**: A CNN model architecture is defined with two convolutional layers, max-pooling, fully connected layers, and dropout for regularization.
  3. **Load Data**: FashionMNIST dataset is loaded for training and validation sets. Data transformations convert images to tensors.
  4. **Instantiate Components**: The CNN model, criterion (CrossEntropyLoss), and optimizer (SGD) are instantiated for training.
  5. **Train Model**: The model is trained over multiple epochs using the training data, computing loss and accuracy metrics.
  6. **Evaluate Model**: The trained model is evaluated on the validation set, generating a classification report and confusion matrix.
  7. **Visualize Results**: Training metrics (loss and accuracy) over epochs are plotted using Matplotlib. Confusion matrix is visualized.
  8. **Export Trained Model**: The trained model is saved to the 'models' folder for future use.

### FashionMNIST_TensorFlow.ipynb
- This notebook implements image classification using a CNN in TensorFlow.
- **Key Steps**:
  1. **Import Libraries**: Libraries such as NumPy, Matplotlib, TensorFlow, and scikit-learn are imported.
  2. **Load Data**: FashionMNIST dataset is loaded and normalized for training and validation sets.
  3. **Define Model**: A CNN model architecture is defined with convolutional layers, max-pooling, flattening, fully connected layers, and dropout.
  4. **Compile Model**: The model is compiled with stochastic gradient descent optimizer and sparse categorical cross-entropy loss.
  5. **Train Model**: The model is trained over multiple epochs and training metrics (loss and accuracy) are monitored.
  6. **Visualize Training Metrics**: Training and validation loss/accuracy are plotted over epochs.
  7. **Generate Evaluation Reports**: Classification report and confusion matrix are generated using scikit-learn.
  8. **Export Trained Model**: The trained model is saved in the 'models' folder for later use.

### models
- This folder contains the trained models from both PyTorch and TensorFlow implementations.

## Summary
The primary objective of this project was to develop models capable of accurately classifying images from the Fashion MNIST dataset. Both the PyTorch and TensorFlow models achieved over 90% accuracy on the validation set. The detailed implementations and results are provided in the respective Jupyter notebooks.
