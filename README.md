# Image Classification with CIFAR-10 Dataset

## Overview

This project focuses on image classification using the CIFAR-10 dataset. The goal is to build a model capable of classifying images into 10 different categories. The implementation includes both a basic Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) for comparison.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The classes include objects like airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Basic ANN Implementation

The initial approach involves building a simple Artificial Neural Network (ANN) with multiple layers. The model is trained using the CIFAR-10 dataset, and the performance is evaluated. The achieved accuracy is reported at the end of the training epochs.

## CNN Implementation

To enhance classification performance, a Convolutional Neural Network (CNN) is implemented. The CNN architecture includes convolutional layers, batch normalization, max-pooling, and dropout for regularization. The training process and evaluation metrics are discussed.

## Model Evaluation

Both ANN and CNN models are evaluated using test sets. The accuracy, loss, and other relevant metrics are visualized to assess model performance. The CNN model significantly outperforms the basic ANN, showcasing the power of convolutional networks in image classification tasks.

## Model Comparison

- **Basic ANN:**
  - Achieved Accuracy: 49% after 5 epochs.

- **CNN:**
  - Achieved Accuracy: 85% after 5 epochs.
  - Improved accuracy and reduced computational complexity compared to ANN.

## Instructions

To run the code locally, follow these steps:

1. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the main script:

    ```bash
    python image_classification.py
    ```

## Dependencies

- TensorFlow
- Keras
- Matplotlib
- NumPy

## Model Storage

The trained CNN model is saved for future use:

```bash
model.save('image_classification_CNN.h5')

## Acknowledgements
- The project leverages the CIFAR-10 dataset for image classification.
- Both ANN and CNN architectures are implemented for comparison.
- Model evaluation and comparison metrics highlight the advantages of CNNs for image classification.
