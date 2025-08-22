//
//  MachineLearningModel.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/13/25.
//

import Foundation

@MachineLearningModelActor
protocol MachineLearningModel: Sendable {
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

extension MachineLearningModel {
    func category(of vector: Vector<Float>) -> Int {
        vector.maxValueAndIndex().index
    }

    static func heStd(fanIn: Int) -> Float {
        sqrt(2.0 / Float(fanIn))
    }
}

@globalActor
actor MachineLearningModelActor {
    static let shared = MachineLearningModelActor()
    private init() { }
}
