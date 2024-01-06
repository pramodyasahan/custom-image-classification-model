# Custom Image Classification Model

## Overview
This repository hosts a TensorFlow-based custom image classification model. The model is designed to classify images into binary classes, labeled as 'Happy' and 'Sad'.

## Data Preparation
- The images are sourced from a directory named 'data'.
- The dataset is split into training (70%), validation (20%), and testing (10%) sets.
- Images are resized to 256x256 pixels and normalized to have pixel values in the range [0, 1].

## Model Architecture
The Sequential model consists of:
- Convolutional layers (Conv2D) with ReLU activation for feature extraction.
- MaxPooling layers for downsampling.
- A Flatten layer to convert 2D features into a 1D vector.
- Dense layers with ReLU and sigmoid activations for classification.

## Compilation and Training
- The model is compiled using the Adam optimizer and binary crossentropy loss function.
- Accuracy is used as a metric.
- Training occurs over 20 epochs with validation data for performance monitoring.
- TensorBoard is used for tracking and visualizing metrics.

## Performance Evaluation
After training, the model's performance is evaluated using:
- Precision
- Recall
- Binary Accuracy
These metrics are calculated on the test dataset.

## Prediction
- The model predicts the class of a new image (e.g., 'cat.jpg').
- The image is resized to 256x256 pixels, normalized, and fed into the model for prediction.
- The output classifies the image as either 'Happy' or 'Sad'.

## Visualization
- Loss and validation loss over epochs are plotted using Matplotlib.
- The original and resized images are displayed using Matplotlib.

## Usage
To use the model:
1. Prepare a dataset in a directory and load it using `tf.keras.utils.image_dataset_from_directory`.
2. Split the dataset into training, validation, and testing sets.
3. Define and compile the Sequential model.
4. Train the model using the training data and validate it.
5. Evaluate the model using precision, recall, and accuracy metrics.
6. Predict the class of new images.

## Dependencies
- numpy
- tensorflow
- matplotlib
- os (for file handling)
- cv2 (for image processing)

## Note
This code is designed for binary image classification and can be adapted for other similar tasks. The model's architecture, hyperparameters, and training duration can be modified to suit different datasets and requirements.
