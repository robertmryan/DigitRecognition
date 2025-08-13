//
//  MachineLearningModel.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/13/25.
//

protocol MachineLearningModel {
    /// Train model for one input.
    ///
    /// - Parameters:
    ///   - x: An input vector.
    ///   - t: Output vector.
    ///   - learningRate: Learning rate.

    func train(x: Vector<Float>, t: Vector<Float>)

    /// Perform inference for one input.
    ///
    /// - Parameter x: An input vector.
    /// - Returns: An output vector.

    func inference(of x: Vector<Float>) -> Vector<Float>
}
