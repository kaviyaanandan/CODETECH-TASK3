Name: KAVIYA

Company: COOTECH IT SOLUTIONS
ID: CT04DR2957
Domain: Machine Learning
Duration: DEC 2025 – JAN 2026
Mentor: MUZAMMIL


Overview of the Project
Project Title- Image Classification Model Using Convolutional Neural Network (CNN)

Objective

The objective of this project is to build an Image Classification model using a Convolutional Neural Network (CNN) with TensorFlow and Keras. The model is trained on the CIFAR-10 dataset, which contains images from 10 different classes.

This project aims to understand how CNNs extract visual features, learn spatial patterns in images, and perform multi-class classification. It also focuses on evaluating the performance of the trained model using test data.

Key Activities
· Dataset Loading

The CIFAR-10 dataset is loaded using TensorFlow’s built-in dataset API. The dataset consists of:

60,000 color images (32×32 pixels)

10 image categories

Separate training and testing datasets

· Data Preprocessing

Image data is normalized by scaling pixel values to the range 0–1. This improves training stability and speeds up model convergence.

· Model Architecture Design

A Sequential CNN model is designed using:

Convolutional layers for feature extraction

MaxPooling layers for dimensionality reduction

Fully connected (Dense) layers for classification

Softmax output layer for multi-class prediction

· Model Compilation

The model is compiled using:

Adam optimizer

Sparse Categorical Crossentropy loss function

Accuracy as the evaluation metric

· Model Training

The CNN model is trained on the training dataset for multiple epochs with a fixed batch size. Validation is performed using the test dataset to monitor performance during training.

· Model Evaluation

The trained model is evaluated on the test dataset to measure classification accuracy and overall performance.
