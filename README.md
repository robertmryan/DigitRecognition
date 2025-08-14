# MNIST Digit Recognition Demo

This is a demonstration of MNIST digit recognition using a neural network.

This is not an attempt at trying to achieve the optimal results, but rather:

 * A simplistic single-layer model using SDG training;
 * Using legacy training and testing datasets found [online](https://github.com/cvdfoundation/mnist?tab=readme-ov-file#mnist); and
 * I deliberately am not availing myself of machine learning frameworks. I am using my own `Matrix` and `Vector` types (which achieve their efficiency through the Accelerate framework’s vDSP and cBLAS). The idea is to really get my arms around the underlying math. Various machine learning libraries are wonderful, but they sometimes abstract you away from the underlying algorithms),

It features:

 * Loading training traditional MNIST dataset of 60,000 images from IDX files and training the model. (Obviously, we frequently train machine learning models offline and then only use the app for inference, but this dataset is sufficiently small that we can do both training, testing, and inference on-device. Training takes less than a second at runtime on modern hardware.)
 * Loading testing traditional dataset of 10,000 images to test the effectiveness of the training.
 * After either loading or training, press right and left buttons to scroll through the dataset, visualizing (a) a blown up rendition of the image on the left; and (b) the categorization of the inference in a bar chart on the right.

Developed in Xcode 16.4 running Swift 6 (Swift 6.1.2).

- - -

[Copyright © 2025 Robert M. Ryan](LICENSE.md)
