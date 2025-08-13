//
//  SGDSingleLayer.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/9/25.
//

import Foundation

/// Stochastic Gradient Descent.

struct SGDSingleLayer: MachineLearningModel {
    /// The single layer matrix underpinning this model
    private var w: Matrix<Float>

    /// The model’s current bias vector
    private var b: Vector<Float>

    /// The model’s learning rate
    let learningRate: Float = 0.01

    init(inputCount: Int, outputCount: Int) {
        w = Matrix<Float>(repeating: 0.01, rows: inputCount, cols: outputCount)
        b = Vector<Float>(repeating: 0, count: inputCount)
    }
}

extension SGDSingleLayer {
    func train(
        x: Vector<Float>,       // Input vector (784,)
        t: Vector<Float>,       // Target one-hot vector (10,)
    ) {
        let numClasses = t.count
        let inputSize = x.count

        // 1. Compute logits z = W * x + b
        // z shape: (10,)
        let z = w.multipliedBy(x, plus: b)

        // 2. Compute softmax probabilities y
        let y = z.softmax()

        // 3. Compute error (y - t)
        let error = y - t

        // 4. Compute gradient for W: grad_W = error * x^T
        // grad_W shape: (10, 784)
        // We'll do outer product of error (10,) and x (784,)
        // Update weights: W = W - learningRate * grad_W
        for i in 0..<numClasses {
            let gradRowStart = i * inputSize
            let gradRowStartAddress = w.buffer.baseAddress!.advanced(by: gradRowStart)
            // grad_W[i, :] = error[i] * x[:]
            let scaledX = x * error[i]

            // W[i, :] -= learningRate * grad_W[i, :]
            scaledX.multiplied(by: -learningRate, plus: gradRowStartAddress)
        }

        // 5. Update bias: b = b - learningRate * error
        error.multiplied(by: -learningRate, plus: b.buffer.baseAddress!)
    }

    func inference(of x: Vector<Float>) -> Vector<Float> {
        w.multipliedBy(x, plus: b).softmax()
    }
}
