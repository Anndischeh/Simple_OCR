# Simple_OCR
This repository contains a simple OCR (Optical Character Recognition) system designed to recognize English handwritten letters and digits.

## Introduction 
Within this repository lies a straightforward OCR system intended for identifying English letters and digits. It operates based on the [MNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset), which provides a collection of handwritten character digits derived from the NIST Special Database 19. The dataset is formatted into 28x28 pixel images, aligning with the structure of the MNIST dataset. To prepare the data, we employ a transformer class called Reshaper, which adjusts the input data into a specified format. Data preprocessing for training and testing involves tasks such as assigning column names, resetting indices, extracting features and labels, and applying a pipeline consisting of standard scaling and custom reshaping.

## Method 
Our approach involves developing three distinct Convolutional Neural Network (CNN) models tailored for image classification tasks. Each model is constructed using the Keras Sequential API and comprises convolutional layers, max-pooling layers, flattening layers, and fully connected layers. Model 1 incorporates one dense layer, Model 2 integrates two dense layers with dropout regularization, and Model 3 encompasses three dense layers with dropout regularization. All models yield probabilities for classification into 62 classes using softmax activation. Below is a comparative summary of the models:

Model 1:

| Layer  | Output Shape      |
|--------|-------------------|
| Conv2D | (None, 26, 26, 32)|
| MaxPooling2D | (None, 13, 13, 32)|
| Flatten | (None, 5408)      |
| Dense  | (None, 128)       |
| Dense  | (None, 62)        |

Model 2:

| Layer  | Output Shape      |
|--------|-------------------|
| Conv2D | (None, 26, 26, 32)|
| Dropout| (None, 26, 26, 32)|
| MaxPooling2D | (None, 13, 13, 32)|
| Flatten | (None, 5408)      |
| Dense  | (None, 128)       |
| Dropout| (None, 128)       |
| Dense  | (None, 62)        |

Model 3:

| Layer  | Output Shape      |
|--------|-------------------|
| Conv2D | (None, 26, 26, 32)|
| Dropout| (None, 26, 26, 32)|
| MaxPooling2D | (None, 13, 13, 32)|
| Flatten | (None, 5408)      |
| Dense  | (None, 256)       |
| Dropout| (None, 256)       |
| Dense  | (None, 128)       |
| Dense  | (None, 62)        |


## Result 
All three models underwent training for 5 epochs. The results and the process are visualized in the plot below. While the performance of all models is roughly similar, Model 3 exhibits slightly higher accuracy.

![Training Results](https://github.com/Anndischeh/Simple_OCR/blob/e46c9a403936a6084b5d4adfa59a62759d34b2a1/media/result.png)

# Example
Here is a demonstration of the model's performance on a sample of test data:

![Sample Test Data](https://github.com/Anndischeh/Simple_OCR/blob/e46c9a403936a6084b5d4adfa59a62759d34b2a1/media/samples.png)
