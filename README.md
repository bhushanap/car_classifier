# 🚗 Car Model Classification with Stanford Cars Dataset

## 📘 Introduction

![image](https://github.com/bhushanap/car_classifier/assets/83635464/bcc306aa-a86a-4d36-9b33-a806affaae54)

The Stanford Cars Dataset is a benchmark in the field, offering a diverse collection of over 16,000 car images spanning 196 different classes, each representing a unique car model. This rich dataset not only poses a significant challenge due to the diversity in vehicle appearances but also provides an excellent opportunity to develop and fine-tune machine learning models for accurate classification.

## 🎯 Goals

The primary objective was to design a robust and efficient car model classifier that can accurately identify and categorize car models from images.

This was done using a ResNet34 architecture, a robust convolutional neural network (CNN), as it can generalize well to large datasets by learning the residual differences in each new layer.

![image](https://github.com/bhushanap/car_classifier/assets/83635464/1fc11b83-a5cf-4073-af2a-8ca6feaa563b)

## 🛠️ Work Done

In this project, the Fastai library was used for developing a car model classifier using the Stanford Cars Dataset. The initial steps involved the preparation of dataset loaders specifically tailored for the Stanford Cars Dataset, ensuring seamless loading of both training and testing data. Initially, I was contemplating the implementation of a distinct regression output layer for car make prediction, but then I went ahead with a single classification layer for all classes separately to avoid preprocessing complexities. Instead, all labels were one-hot encoded, paving the way for a more streamlined classification process.

Then I imported the ResNet34 architecture to serve as the backbone for the model. To facilitate fine-tuning, the last layers of ResNet34 were modified to align with the classification task at hand. The process began with an initial fine-tuning stage where the learning rate was dynamically determined using the "get learning rate" step. The model was fine-tuned for three epochs, with all layers frozen except the last.

Following this, another learning rate determination step was executed, leading to a subsequent fine-tuning phase spanning five epochs in which all layers were unfrozen, leading to a 70% accuracy for the car model classification task.

![image](https://github.com/bhushanap/car_classifier/assets/83635464/01618667-bf1e-4b14-8262-6554fcfbb3e0)

## 📊 Results

Some of the images in the test set were challenging even for a human to distinguish easily. Even then, the model performed pretty well!

Even though only 70% accuracy was achieved, the wrong predictions were of cars/models pretty similar to the actual car.

A Hugging Face demo of the model running can be seen here. It should be noted that the model will only output classes and models in the dataset. So if there is no image of a particular car in the Stanford image dataset, it won't work properly.

![results](demo.gif)
