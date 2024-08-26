# face expression detector
 ## Project Overview
This project focuses on emotion classification from facial expressions using deep convolutional neural networks (CNNs). The model was trained on the FER-2013 dataset, a widely recognized benchmark introduced at the International Conference on Machine Learning (ICML). The dataset comprises 35,887 grayscale images, each with a resolution of 48x48 pixels, categorized into seven distinct emotions: anger, disgust, fear, happiness, neutrality, sadness, and surprise.
 ## Dependencies
 * Python
 * OpenCV
 * Tensorflow
## Data Preparation
* The original data set is taken from Kaggle and is called [the fer2013 dataset](https://www.kaggle.com/datasets/deadskull7/fer2013). I converted the CSV data set into PNG format.
* The code for data preprocessing in the <sup> Data.ipynb </sup> file.
## Algorithm
* This implementation by default detects the emotions on all faces in the primary camera feed with a simple four-layer CNN.
* The Haar Cascade method detects each frame of the primary camera feed.
* The region of the image containing the face is resized to 48*48 and is passed as input to the Computational Neural Network (CNN).
* The Network outputs a list of softmax scores for the different classes,
* The class with the maximum score is displayed on the screen.
* The accuracy reached around 60% in 50 epochs.
* Here is the plot of the accuracy and loss of each validation and training data.
![](https://github.com/Sangeeth1242/face_expression_detection/blob/main/plot.png)
