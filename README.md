# MNIST Digit Recognition Demo

This is a demonstration of MNIST digit recognition using a neural network.

This is not an attempt at trying to achieve the optimal results, but rather:

 * A simple model:
     * Simplistic single-layer linear model
     * Stochastic Gradient Descent (SGD) weight adjustment algorithm
     * Softmax activation to produce category probabilities
     * Cross-entropy loss function
 * Using legacy training and testing datasets found [online](https://github.com/cvdfoundation/mnist?tab=readme-ov-file#mnist); and
 * This is deliberately *not* availing itself of machine learning frameworks. To fully appreciate the mathmatics underpinning the model, this uses `Matrix` and `Vector` types (but using the Accelerate framework’s, notably vDSP and cBLAS, to improve performance). The idea is to really appreciate the implementaton details of the model. Various machine learning libraries are wonderful, but they abstract you away from the underlying algorithms.

It features:

 * Loading training traditional MNIST dataset of 60,000 images from IDX files and training the model. (Obviously, we frequently train machine learning models offline and then only use the app for inference, but this dataset is sufficiently small that we can do both training, testing, and inference on-device. Training takes less than a second at runtime on modern hardware.)
 * Loading testing traditional dataset of 10,000 images to test the effectiveness of the training.
 * After either loading or training, press right and left buttons to scroll through the dataset, visualizing (a) a blown up rendition of the image on the left; and (b) the categorization of the inference in a bar chart on the right.
 * Once trained, you can draw on the 28×28 grid and tap the “Process” button and it will show you the inference results (as a probability of which category the image falls).

Open items:

 * When drawing your own handwritten character for recognition, it is highly dependent based upon where you drew the character. The MNIST pipeline for inputs apparently entails rendering the handwritten character as a 20×20 image that has been geometrically-centered (!) within a 28×28 image. The conversion of our handwritten characters should adopt the same pipeline to get meaningful benchmark for how well the model recognizes our handwriting.
 * We should show the % accuracy for the test dataset after performing inference.
 * It might be nice to have a filter for mis-categorized images, to be able to focus on the edge cases.
 * Implement one, maybe two, hidden ReLU layers, and see how that improves accuracy. Also consider a CNN model.
 * Identify possibly better datasets. E.g., the original MNIST dataset used here performs notoriously badly with “4”s (as the dataset has a serious underrepresentation of “4”s that are closed at the top) and “7”s.

Developed in Xcode 16.4 running Swift 6 (Swift 6.1.2).

- - -

[Copyright © 2025 Robert M. Ryan](LICENSE.md)
