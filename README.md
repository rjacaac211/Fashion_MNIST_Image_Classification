# Fashion MNIST Image Classification

## Overview
This project implements image classification on the Fashion MNIST dataset using a Convolutional Neural Network (CNN) in PyTorch. The Fashion MNIST dataset comprises 70,000 images of fashion items categorized into 10 classes.

## Contents

### FashionMNIST_PyTorch.ipynb
- Import libraries: Import torch, torch.nn, numpy, matplotlib.pyplot, torchvision.datasets, transforms, torch.utils.data, sklearn.metrics.
- Define model: Define a CNN model with two convolutional layers followed by max pooling, two fully connected layers, and dropout for regularization.
- Load data: Load FashionMNIST dataset for both training and validation sets, applying transformations to convert images to tensors.
- Instantiate components: Instantiate the CNN model, criterion (CrossEntropyLoss), and optimizer (SGD) for training.
- Train model: Train the model over multiple epochs using the training data, computing loss and accuracy metrics.
- Evaluate model: Evaluate the trained model on the validation set, generating a classification report and confusion matrix using sklearn.metrics.
- Visualize results: Plot training metrics (loss and accuracy) over epochs using Matplotlib and visualize the confusion matrix.
- Save model: Save the trained model to the 'models' folder for future use.

### models
- This folder contains the trained model/s in this project.

## Summary
The primary objective of this project was to develop a model capable of accurately classifying images of the Fashion MNIST dataset. The model achieved a classification accuracy of 92%.
