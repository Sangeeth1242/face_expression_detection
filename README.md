# Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Model Explanation](#model-explanation)
5. [Model Workflow](#model-workflow)
6. [Conclusion](#conclusion)

## Project Overview

This project focuses on emotion classification from facial expressions using deep convolutional neural networks (CNNs). The model was trained on the FER-2013 dataset, a widely recognized benchmark introduced at the International Conference on Machine Learning (ICML). The dataset comprises 35,887 grayscale images, each with a resolution of 48x48 pixels, categorized into seven distinct emotions: anger, disgust, fear, happiness, neutrality, sadness, and surprise. This project showcases the ability of CNNs to learn complex features from raw image data for classification tasks.

## Dataset

The **FER2013** dataset is used for training and testing the model. The dataset can be downloaded from Kaggle:

- [FER2013 Kaggle Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

The dataset is structured into two main directories:
- **train**: Contains labeled images for training the model.
- **test**: Contains images for testing the model after training.

Each directory has subfolders for each of the 7 facial expressions: **Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, and **Neutral**.


## Installation

### Prerequisites

Before running the project, you need to install the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- Numpy
- OpenCV
- Matplotlib
- argparse

You can install the required dependencies using `pip`:

```bash
pip install tensorflow numpy opencv-python matplotlib
```
## Model Explanation

The model is built using Convolutional Neural Networks (CNN), which are highly effective for image classification tasks. The architecture consists of several layers:

1. **Convolutional Layers**: These layers are responsible for feature extraction from the input images. They apply filters to the image to detect edges, textures, and patterns.
2. **MaxPooling Layers**: These layers reduce the spatial dimensions of the input, which helps in reducing computation and avoiding overfitting.
3. **Dropout Layers**: Dropout is applied to prevent overfitting by randomly disabling neurons during training.
4. **Fully Connected (Dense) Layers**: These layers help in combining the features extracted from the convolutional layers and making the final classification decision.
5. **Output Layer**: The output layer consists of 7 neurons corresponding to the 7 facial expressions, with a softmax activation function that outputs a probability distribution.

The model is trained on the FER2013 dataset using image augmentation techniques to improve generalization. The model is evaluated based on its accuracy and loss over the training and testing datasets.

## Model Workflow

- **Image Preprocessing**: Images are resized and normalized to fit the input size of the network (e.g., 48x48 pixels).
- **Training**: The model is trained using categorical cross-entropy loss and the Adam optimizer. The training process involves splitting the dataset into training and validation sets, and monitoring the model's performance during each epoch.
- **Evaluation**: After training, the model is evaluated using a test set to estimate its real-world performance.
- Here is the plot of the accuracy and loss of each validation and training data.
![](https://github.com/Sangeeth1242/face_expression_detection/blob/main/plot.png)

## Conclusion

This project provides a solution for facial expression recognition using a Convolutional Neural Network. By leveraging the FER2013 dataset, the model is able to accurately classify facial expressions into seven categories. This system can be utilized in a wide range of applications, including enhancing user experience, security monitoring, and psychological studies. Future improvements could include expanding the dataset, fine-tuning the model for better accuracy, and integrating it into real-time systems for emotion detection.

