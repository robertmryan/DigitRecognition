//
//  SGDTwoHiddenLayer.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/21/25.
//

import Foundation

/// Two-hidden-layer MLP with Stochastic Gradient Descent.
/// Layout assumption:
///   - Each weight matrix has shape (out, in).
///   - Forward uses: z = W * a_prev + b, where each row of W is dotted with a_prev.
///   - Updates write per-output row with a scaled copy of the input activation.

final class SGDTwoHiddenLayer: MachineLearningModel {
    // Layer sizes
    let inputSize: Int
    let hidden1: Int
    let hidden2: Int
    let outputSize: Int

    // Weights and biases: shapes (out, in) and (out)
    private let W1: Matrix<Float>
    private let b1: Vector<Float>

    private let W2: Matrix<Float>
    private let b2: Vector<Float>

    private let W3: Matrix<Float>
    private let b3: Vector<Float>

    // Hyperparameters
    private let learningRate: Float

    init(inputVectorSize: Int, hidden1: Int = 512, hidden2: Int = 256, outputVectorSize: Int, learningRate: Float = 0.01) {
        self.inputSize   = inputVectorSize
        self.hidden1     = hidden1
        self.hidden2     = hidden2
        self.outputSize  = outputVectorSize
        self.learningRate = learningRate

        // Simple small random-ish init (use your preferred RNG/init); 0.01 constant like your example:

        let heStd1 = Self.heStd(fanIn: inputVectorSize)
        let values1 = (0 ..< hidden1 * inputVectorSize).map { _ in Float.random(in: -1...1) * heStd1 }
        W1 = Matrix<Float>(elements: values1, rows: hidden1, cols: inputVectorSize)
        b1 = Vector<Float>(repeating: 0, count: hidden1)

        let heStd2 = Self.heStd(fanIn: hidden1)
        let values2 = (0 ..< hidden2 * hidden1).map { _ in Float.random(in: -1...1) * heStd2 }
        W2 = Matrix<Float>(elements: values2, rows: hidden2, cols: hidden1)
        b2 = Vector<Float>(repeating: 0, count: hidden2)

        let heStd3 = Self.heStd(fanIn: hidden2)
        let values3 = (0 ..< outputVectorSize * hidden2).map { _ in Float.random(in: -1...1) * heStd3 }
        W3 = Matrix<Float>(elements: values3, rows: outputVectorSize, cols: hidden2)
        b3 = Vector<Float>(repeating: 0, count: outputSize)
    }
}

// MARK: - MachineLearningModel interface

extension SGDTwoHiddenLayer {
    /// One-sample SGD step with softmax + cross-entropy gradient (delta3 = y - t).
    func train(x: Vector<Float>, t: Vector<Float>) {
        // ---- Forward ----
        let z1 = W1.multiplied(by: x,  plus: b1)
        let a1 = z1.relu()
        let z2 = W2.multiplied(by: a1, plus: b2)
        let a2 = z2.relu()
        let z3 = W3.multiplied(by: a2, plus: b3)
        let y  = z3.softmax()

        // ---- Backward ----
        // Output layer gradient (softmax+CE): delta3 = y - t
        let delta3 = y - t                           // (output)

        // Hidden2 gradient: delta2 = (W3^T * delta3) ⊙ relu'(z2)
        let delta2 = W3.transposeMultiply(delta3)    // (hidden2)
        delta2.formMultiplyInPlace(by: z2.reluPrime())

        // Hidden1 gradient: delta1 = (W2^T * delta2) ⊙ relu'(z1)
        let delta1 = W2.transposeMultiply(delta2)    // (hidden1)
        delta1.formMultiplyInPlace(by: z1.reluPrime())

        // ---- SGD Updates ----
        // W3 rows update: W3[j, :] -= lr * (delta3[j] * a2[:])
        W3.rowWiseUpdate(delta: delta3, prevAct: a2, learningRate: learningRate)
        // b3 -= lr * delta3
        delta3.multiplied(by: -learningRate, plus: b3.buffer.baseAddress!)

        // W2 rows update
        W2.rowWiseUpdate(delta: delta2, prevAct: a1, learningRate: learningRate)
        // b2 -= lr * delta2
        delta2.multiplied(by: -learningRate, plus: b2.buffer.baseAddress!)

        // W1 rows update
        W1.rowWiseUpdate(delta: delta1, prevAct: x,  learningRate: learningRate)
        // b1 -= lr * delta1
        delta1.multiplied(by: -learningRate, plus: b1.buffer.baseAddress!)
    }

    func inference(of x: Vector<Float>) -> Vector<Float> {
        let a1 = W1.multiplied(by: x,  plus: b1).relu()
        let a2 = W2.multiplied(by: a1, plus: b2).relu()
        return W3.multiplied(by: a2, plus: b3).softmax()
    }
}
