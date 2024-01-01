# Fashion MNIST Classification with Convolutional Neural Network

## Introduction

This project focuses on building and training a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The dataset consists of 70,000 grayscale images of 10 fashion categories, with each image being 28x28 pixels. Our goal is to create a model that can accurately predict the category of a given fashion item.

## Prerequisites

Before executing the code, ensure the following software and libraries are installed on your local machine:

- Python 3
- TensorFlow 2.x
- Jupyter Notebook
- Required Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

If you do not have these components, please install them before proceeding to the execution steps.

## Installation

1. **Install Python 3:** Download the latest version of Python 3 from the official Python website (https://www.python.org) based on your operating system and follow the installation instructions.

2. **Install Jupyter Notebook:** Open a command prompt or terminal and run the following command: `pip install jupyter`.

3. **Install Required Libraries:** Execute the following command to install necessary libraries: `pip install numpy pandas matplotlib seaborn scikit-learn tensorflow`.

## Data File

The Fashion MNIST dataset is available directly in TensorFlow. You can load it using the following commands:

```python
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
