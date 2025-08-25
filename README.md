# MNIST Digit Recognition Demo

This is a demonstration of MNIST digit recognition using a neural network.

This is not an attempt at trying to achieve the optimal results, but rather:

 * The ML models:
     * Single Layer Perceptron:
         * Trained on-device when you load “Training Data Set”; training performs single epoch, but you can tap this button several (5-10) times to improve results
         * Stochastic Gradient Descent (SGD) weight adjustment algorithm
         * Softmax activation to produce category probabilities
         * Cross-entropy loss function
         * Roughly 90% accurate
     * Multiple Layer Perceptron, same as above but:
         * Include two hidden ReLU layers (512 and 256, respectively); if you want multiple epochs, it might call for 10-40 runs before you hit diminishing returns
         * Roughly 96% accurate
     * Convolution Neural Network:
         * This is pretrained (in PyTorch), 12 epochs; training took ~80 seconds on M1 Mac
         * Converted to a CoreML model via coremltools
         * Roughly 99.6% accurate
 * Using legacy training and testing datasets found [online](https://github.com/cvdfoundation/mnist?tab=readme-ov-file#mnist); and
 * The SLP/MLP models are deliberately *not* availing itself of machine learning frameworks. To fully appreciate the mathmatics underpinning the model, this uses `Matrix` and `Vector` types (but using the Accelerate framework’s, notably vDSP and cBLAS, to improve performance). The idea is to really appreciate the implementaton details of the model. Various machine learning libraries are wonderful, but they abstract you away from the underlying algorithms. But as a result, we are not also fully optimized. But this is sufficient for illustrative purposes, still fast enough to perform on-device training of the 60,000 image dataset.

It features:

 * Loading training traditional MNIST dataset of 60,000 images from IDX files and training the model. (Obviously, we frequently train machine learning models offline and then only use the app for inference, but this dataset is sufficiently small that we can do both training, testing, and inference on-device. Training takes less than a second at runtime on modern hardware.)
 * Loading testing traditional dataset of 10,000 images to test the effectiveness of the training.
 * After either loading or training, press right and left buttons to scroll through the dataset, visualizing (a) a blown up rendition of the image on the left; and (b) the categorization of the inference in a bar chart on the right.
 * Once trained, you can draw on the 28×28 grid and tap the “Process” button and it will show you the inference results (as a probability of which category the image falls).
 * When drawing your own handwritten character for recognition, it will translate the drawn figure for geometric center of the stroke(s). The MNIST pipeline for inputs apparently entails rendering the handwritten character as a 20×20 image that has been geometrically-centered (!) within a 28×28 image. The conversion of our handwritten characters employs a similar pipeline to get meaningful benchmark for how well the model recognizes our handwriting.

Open items:

 * Identify possibly better datasets. E.g., the original MNIST dataset used here performs notoriously badly with “4”s (as the dataset has a serious underrepresentation of “4”s that are closed at the top) and “7”s.

Developed in Xcode 16.4 running Swift 6 (Swift 6.1.2).

- - -

[Copyright © 2025 Robert M. Ryan](LICENSE.md)
