# MNIST Handwritten Digit Classification

## Objective

The objective of this project is to develop a neural network model that accurately classifies handwritten digits from the MNIST dataset. By leveraging machine learning techniques and computer vision, our aim is to create a robust model capable of automatic recognition and categorization of these digits. The MNIST dataset, despite its simplicity, serves as a fundamental introduction to computer vision.

## Dataset Description

The MNIST dataset is widely recognized and extensively used in machine learning, particularly for image classification tasks. It consists of 60,000 training images and 10,000 testing images, each representing a grayscale handwritten digit of size 28x28 pixels.

![MNIST Dataset Sample](images/data_sample.png)

## Directory Structure

The repository follows a specific directory structure:

### src/model.py

The `model.py` file contains the implementation of the neural network model architecture. The current architecture has been chosen to support the objectives of the project. Here is a summary of the model's layers and parameters:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```

### src/utils.py

This `utils.py` includes helper functions for various tasks such as obtaining the device, training and testing the model, plotting training metrics, and visualizing correct and incorrect predictions made by the model.

### S5.ipynb

The main file for this project is the Jupyter Notebook file, `S5.ipynb`. It encompasses the training and evaluation of the model.

## Results

Our trained model has demonstrated exceptional performance, achieving high accuracy even on challenging images. Here are some examples of correctly classified images:

![Correctly Classified Images](images/correct_classification.png)

However, it is important to note that there are instances where our model struggles to make accurate predictions. The following images present significant challenges for comprehension:

![Incorrectly Classified Images](images/incorrect_classification.png)

These instances serve as a reminder that even state-of-the-art models have limitations when confronted with highly ambiguous or complex images.

Please be aware that the images provided above are representative samples and may not reflect the overall accuracy of the model across the entire MNIST dataset.

For further details on the project and its implementation, refer to the accompanying code and documentation.

**Note:** The images used in the examples are for illustrative purposes and may not accurately reflect the actual performance of the model.