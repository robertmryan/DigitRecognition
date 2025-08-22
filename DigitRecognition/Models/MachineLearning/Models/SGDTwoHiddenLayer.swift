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

// final class SGDTwoHiddenLayer: MachineLearningModel {
//     // Layer sizes
//     let inputSize: Int
//     let hidden1: Int
//     let hidden2: Int
//     let outputSize: Int
//
//     // Weights and biases: shapes (out, in) and (out)
//     private let W1: Matrix<Float>
//     private let b1: Vector<Float>
//
//     private let W2: Matrix<Float>
//     private let b2: Vector<Float>
//
//     private let W3: Matrix<Float>
//     private let b3: Vector<Float>
//
//     // Hyperparameters
//     let learningRate: Float
//
//     /// Create model.
//     ///
//     /// - Parameters:
//     ///   - inputVectorSize: 748 for MNIST.
//     ///   - hidden1: Values between 256 and 512 work well.
//     ///   - hidden2: Values between 128 and 256 work well.
//     ///   - outputVectorSize: 10 for MNIST.
//     ///   - learningRate: 0.01 works well.
//
//     init(inputVectorSize: Int, hidden1: Int = 512, hidden2: Int = 256, outputVectorSize: Int, learningRate: // Float = 0.01) {
//         self.inputSize    = inputVectorSize
//         self.hidden1      = hidden1
//         self.hidden2      = hidden2
//         self.outputSize   = outputVectorSize
//         self.learningRate = learningRate
//
//         // Simple small random-ish init (use your preferred RNG/init); 0.01 constant like your example:
//         W1 = Matrix<Float>(repeating: 0.01, rows: hidden1,          cols: inputVectorSize)
//         b1 = Vector<Float>(repeating: 0,    count: hidden1)
//
//         W2 = Matrix<Float>(repeating: 0.01, rows: hidden2,          cols: hidden1)
//         b2 = Vector<Float>(repeating: 0,    count: hidden2)
//
//         W3 = Matrix<Float>(repeating: 0.01, rows: outputVectorSize, cols: hidden2)
//         b3 = Vector<Float>(repeating: 0,    count: outputVectorSize)
//     }
// }
//
// // MARK: - Training / Inference
//
// extension SGDTwoHiddenLayer {
//     /// One-sample SGD step with softmax + cross-entropy gradient (delta3 = y - t).
//     func train(x: Vector<Float>, t: Vector<Float>) {
//         // ---- Forward ----
//         let z1 = W1.multiplied(by: x,  plus: b1)     // (hidden1)
//         let a1 = z1.relu()
//         let z2 = W2.multiplied(by: a1, plus: b2)     // (hidden2)
//         let a2 = z2.relu()
//         let z3 = W3.multiplied(by: a2, plus: b3)     // (output)
//         let y  = z3.softmax()                        // (output)
//
//         // ---- Backward ----
//         // Output layer gradient (softmax+CE): delta3 = y - t
//         let delta3 = y - t                           // (output)
//
//         // Hidden2 gradient: delta2 = (W3^T * delta3) ⊙ relu'(z2)
//         let delta2 = W3.transposeMultiply(delta3)    // (hidden2)
//         delta2.formMultiplyInPlace(by: z2.reluPrime())
//
//         // Hidden1 gradient: delta1 = (W2^T * delta2) ⊙ relu'(z1)
//         let delta1 = W2.transposeMultiply(delta2)    // (hidden1)
//         delta1.formMultiplyInPlace(by: z1.reluPrime())
//
//         // ---- SGD Updates ----
//         // W3 rows update: W3[j, :] -= lr * (delta3[j] * a2[:])
//         sgdRowWiseUpdate(W: W3, delta: delta3, prevAct: a2, lr: learningRate)
//         // b3 -= lr * delta3
//         delta3.multiplied(by: -learningRate, plus: b3.buffer.baseAddress!)
//
//         // W2 rows update
//         sgdRowWiseUpdate(W: W2, delta: delta2, prevAct: a1, lr: learningRate)
//         // b2 -= lr * delta2
//         delta2.multiplied(by: -learningRate, plus: b2.buffer.baseAddress!)
//
//         // W1 rows update
//         sgdRowWiseUpdate(W: W1, delta: delta1, prevAct: x,  lr: learningRate)
//         // b1 -= lr * delta1
//         delta1.multiplied(by: -learningRate, plus: b1.buffer.baseAddress!)
//     }
//
//     func inference(of x: Vector<Float>) -> Vector<Float> {
//         let a1 = W1.multiplied(by: x,  plus: b1).relu()
//         let a2 = W2.multiplied(by: a1, plus: b2).relu()
//         return W3.multiplied(by: a2, plus: b3).softmax()
//     }
// }
//
// // MARK: - Helpers
//
// private extension SGDTwoHiddenLayer {
//     func sgdRowWiseUpdate(
//         W: Matrix<Float>,
//         delta: Vector<Float>,
//         prevAct: Vector<Float>,
//         lr: Float
//     ) {
//         // For each output unit j, do: W[j, :] -= lr * delta[j] * prevAct[:]
//         let rowLen = prevAct.count
//         for j in 0..<delta.count {
//             let rowStart = j * rowLen
//             let rowPtr = W.buffer.baseAddress!.advanced(by: rowStart)
//             let scaled = prevAct * delta[j]
//             scaled.multiplied(by: -lr, plus: rowPtr)
//         }
//     }
// }
//
// // MARK: - Activations
//
// private extension Vector where Element == Float {
//     func relu() -> Vector<Float> {
//         let out = Vector(self)
//         for i in 0..<out.count { out[i] = Swift.max(0, out[i]) }
//         return out
//     }
//
//     func reluPrime() -> Vector<Float> {
//         let out = Vector(self)
//         for i in 0..<out.count { out[i] = (out[i] > 0) ? 1 : 0 }
//         return out
//     }
//
//     func formMultiplyInPlace(by other: Vector<Float>) {
//         precondition(self.count == other.count)
//         for i in 0..<count { self[i] *= other[i] }
//     }
// }
//
// // MARK: - Minimal transpose-multiply
//
// private extension Matrix where Element == Float {
//     /// Compute v = W^T * u, where W is (out, in), u is (out), result is (in).
//     func transposeMultiply(_ u: Vector<Float>) -> Vector<Float> {
//         let out = self.rows
//         let inn = self.cols
//         precondition(u.count == out)
//
//         let v = Vector<Float>(repeating: 0, count: inn)
//         // Accumulate: for each output row j, v += u[j] * W[j, :]
//         for j in 0..<out {
//             let rowStart = j * inn
//             let wRow = self.buffer.baseAddress!.advanced(by: rowStart)
//             let scale = u[j]
//             for i in 0..<inn {
//                 v[i] += scale * wRow[i]
//             }
//         }
//         return v
//     }
// }

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
    let learningRate: Float

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

// MARK: - Training / Inference

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
        rowWiseUpdate(W: W3, delta: delta3, prevAct: a2, learningRate: learningRate)
        // b3 -= lr * delta3
        delta3.multiplied(by: -learningRate, plus: b3.buffer.baseAddress!)

        // W2 rows update
        rowWiseUpdate(W: W2, delta: delta2, prevAct: a1, learningRate: learningRate)
        // b2 -= lr * delta2
        delta2.multiplied(by: -learningRate, plus: b2.buffer.baseAddress!)

        // W1 rows update
        rowWiseUpdate(W: W1, delta: delta1, prevAct: x,  learningRate: learningRate)
        // b1 -= lr * delta1
        delta1.multiplied(by: -learningRate, plus: b1.buffer.baseAddress!)
    }

    func inference(of x: Vector<Float>) -> Vector<Float> {
        let a1 = W1.multiplied(by: x,  plus: b1).relu()
        let a2 = W2.multiplied(by: a1, plus: b2).relu()
        return W3.multiplied(by: a2, plus: b3).softmax()
    }
}

// MARK: - Helpers

private extension SGDTwoHiddenLayer {
    func rowWiseUpdate(
        W: Matrix<Float>,
        delta: Vector<Float>,
        prevAct: Vector<Float>,
        learningRate: Float
    ) {
        // For each output unit j, do: W[j, :] -= lr * delta[j] * prevAct[:]
        let rowLen = prevAct.count
        for j in 0..<delta.count {
            let rowPtr = W.buffer.baseAddress!.advanced(by: j * rowLen)
            let scaled = prevAct * delta[j]              // NEW temp buffer
            scaled.multiplied(by: -learningRate, plus: rowPtr)     // row -= lr * scaled
        }
    }
}

// MARK: - Activations

private extension Vector where Element == Float {
    /// Returns a NEW vector: out[i] = max(0, self[i])
    func relu() -> Vector<Float> {
        let out = Vector<Float>(repeating: 0, count: count)
        for i in 0..<count { out[i] = self[i] > 0 ? self[i] : 0 }
        return out
    }

    /// Returns a NEW vector of ReLU'(z): 1 if z[i] > 0 else 0
    func reluPrime() -> Vector<Float> {
        let out = Vector<Float>(repeating: 0, count: count)
        for i in 0..<count { out[i] = self[i] > 0 ? 1 : 0 }
        return out
    }

    /// In-place Hadamard product: self[i] *= other[i]
    func formMultiplyInPlace(by other: Vector<Float>) {
        precondition(self.count == other.count)
        for i in 0..<self.count { self[i] *= other[i] }
    }
}

// MARK: - Minimal transpose-multiply

private extension Matrix where Element == Float {
    /// Compute v = W^T * u, where W is (out, in), u is (out), result is (in).
    func transposeMultiply(_ u: Vector<Float>) -> Vector<Float> {
        let out = self.rows
        let inn = self.cols
        precondition(u.count == out)

        let v = Vector<Float>(repeating: 0, count: inn)
        // Accumulate: for each output row j, v += u[j] * W[j, :]
        for j in 0 ..< out {
            let rowStart = j * inn
            let wRow = self.buffer.baseAddress!.advanced(by: rowStart)
            let scale = u[j]
            for i in 0 ..< inn {
                v[i] += scale * wRow[i]
            }
        }
        return v
    }
}
