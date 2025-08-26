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

    func inference(of x: Vector<Float>) async -> Vector<Float>

    var requiresOnDeviceTraining: Bool { get }
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

enum ModelType: CaseIterable, Identifiable {
    case singleLayerPerceptron
    case multiLayerPerceptron
    case convolutionalNeuralNetwork

    var id: Self { self }
}

extension ModelType {
    var shortLabel: String {
        switch self {
            case .singleLayerPerceptron:      "SLP"
            case .multiLayerPerceptron:       "MLP"
            case .convolutionalNeuralNetwork: "CNN"
        }
    }

    var longLabel: String {
        switch self {
            case .singleLayerPerceptron:      "Single Layer Perceptron"
            case .multiLayerPerceptron:       "Multi Layer Perceptron"
            case .convolutionalNeuralNetwork: "Convolutional Neural Network"
        }
    }

    @MachineLearningModelActor
    func model() throws -> MachineLearningModel {
        switch self {
            case .singleLayerPerceptron:       SGDSingleLayer(inputVectorSize: 28 * 28, outputVectorSize: 10)
            case .multiLayerPerceptron:        SGDTwoHiddenLayer(inputVectorSize: 28 * 28, outputVectorSize: 10)
            case .convolutionalNeuralNetwork:  try MNISTSmallCNN()
        }
    }
}
