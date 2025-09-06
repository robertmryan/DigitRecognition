# MNIST Digit Recognition Demo

This is a demonstration of MNIST digit recognition using machine learning.

 * The ML models:
     * Single Layer Perceptron:
         * This is not an attempt at trying to achieve the optimal results, but rather an illustration of the most basic of ML models
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
 * The SLP/MLP models are deliberately *not* availing themselves of any machine learning frameworks. To fully appreciate the mathematics underpinning the model, this uses `Matrix` and `Vector` types (but using the Accelerate frameworks, notably vDSP and cBLAS, to improve performance). The idea is to really appreciate the implementation details of the model. Various machine learning libraries are generally preferred, but they abstract one away from the underlying algorithms. But as a result, the two perceptron models are not as efficient as they might be. But this is sufficient for illustrative purposes; it is still fast enough to perform on-device training of the 60,000 image dataset.

It features:

 * Loading training traditional MNIST dataset of 60,000 images from IDX files and training the model. (Obviously, we frequently train machine learning models offline and then only use the app for inference, but this dataset is sufficiently small that we can do both training, testing, and inference on-device. Training takes less than a second at runtime on modern hardware.)
 * Loading testing traditional dataset of 10,000 images to test the effectiveness of the training.
 * After either loading or training, press right and left buttons to scroll through the dataset, visualizing (a) a blown up rendition of the image on the left; and (b) the categorization of the inference in a bar chart on the right.
 * Once trained, you can draw on the 28×28 grid and tap the “Process” button and it will show you the inference results (as a probability of which category the image falls).
 * When drawing your own handwritten character for recognition, it will translate the drawn figure for geometric center of the stroke(s). The MNIST pipeline for inputs apparently entails rendering the handwritten character as a 20×20 image that has been geometrically-centered (!) within a 28×28 image. The conversion of our handwritten characters employs a similar pipeline to get meaningful benchmark for how well the model recognizes our handwriting.
 * When displaying bar chart of the inference results, they are color coded: Green = inference matches expected output; Red = model inference was incorrect; Blue = inference performed on hand-written digit, but there is no expected category, and thus cannot be classified as “right” or “wrong”.
 * Next to the model-picker, there is an “info” button, that shows a high-level drawing that outlines the basic model; this is not intended to be a precise model, but rather a conceptual representation for a non-technical audience.
 * Support for macOS, iPadOS, and iOS targets.

Open items:

 * Identify possibly better datasets.
     * The original MNIST dataset, used here, performs notoriously badly (with perceptron models, at least) with “4”s (as the dataset has a serious under-representation of “4”s that are closed at the top) and “7”s.

Developed in Xcode 16.4 running Swift 6 (Swift 6.1.2).

- - -

[Copyright © 2025 Robert M. Ryan](LICENSE.md)
