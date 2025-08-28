//
//  MNISTSmallCNN.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/24/25.
//

import Foundation
@preconcurrency import CoreML
import Vision

final class MNISTSmallCNN: MachineLearningModel {
    let requiresOnDeviceTraining = false

    private let model: MLModel
    private let pixelBufferBool = GrayPixelBufferPool()

    init() throws {
        let modelURL = Bundle.main.url(forResource: "MNISTSmallCNN_v2", withExtension: "mlmodelc")!
        model = try MLModel(contentsOf: modelURL)

        let mlModel = try MLModel(contentsOf: modelURL)

        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            print("INPUT:", name, "|", desc.type)
        }

        print("PREDICTED LABEL:",
              mlModel.modelDescription.predictedFeatureName ?? "nil")
        print("PROBABILITIES:",
              mlModel.modelDescription.predictedProbabilitiesName ?? "nil")

        for (name, feat) in mlModel.modelDescription.outputDescriptionsByName {
            switch feat.type {
                case .dictionary:
                    let d = feat.dictionaryConstraint!
                    let key = (d.keyType == .string) ? "String" : "Int64"
                    print("OUTPUT:", name, "| DICTIONARY<\(key), Double>")
                case .multiArray:
                    let m = feat.multiArrayConstraint
                    print("OUTPUT:", name, "| MULTIARRAY", m?.shape ?? [])
                case .string:
                    print("OUTPUT:", name, "| STRING")
                default:
                    print("OUTPUT:", name, "|", feat.type) // other cases rarely used here
            }
        }
    }

    func train(x: Vector<Float>, t: Vector<Float>) {
        // this is intentionally blank … already trained
    }

    func inference(of x: Vector<Float>) async -> Vector<Float> {
        var results = Vector<Float>(repeating: 0, count: 10)

        let bytes = Array(x.buffer).map { UInt8($0 * 255) }
        let pixelBuffer = bytes.withUnsafeBytes { pointer in
            pixelBufferBool.make(pointer)
        }

        let featureValue = MLFeatureValue(pixelBuffer: pixelBuffer)

        let provider = try! MLDictionaryFeatureProvider(dictionary: [
            "image": featureValue
        ])
        let prediction = try! await model.prediction(from: provider)

        guard let probabilities = prediction.featureValue(for: "classLabel_probs")?.dictionaryValue as? [String: NSNumber] else {
            fatalError("Probabilities not found")
        }

        for probability in probabilities {
            if let index = Int(probability.key) {
                let value = probability.value.floatValue
                results[index] = value
            }
        }

        return results
    }

    func inferenceSlow(of x: Vector<Float>) async -> Vector<Float> {
        var results = Vector<Float>(repeating: 0, count: 10)

        let bytes = Array(x.buffer).map { UInt8($0 * 255) }
        guard
            let image = bytes.image(inverting: false)
        else {
            fatalError("Could not construct image")
        }

        let imageConstraint = model.modelDescription.inputDescriptionsByName["image"]!.imageConstraint!
        let provider = try! MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(cgImage: image, constraint: imageConstraint)
        ])
        let prediction = try! await model.prediction(from: provider)

        guard let probabilities = prediction.featureValue(for: "classLabel_probs")?.dictionaryValue as? [String: NSNumber] else {
            fatalError("Probabilities not found")
        }

        for probability in probabilities {
            if let index = Int(probability.key) {
                let value = probability.value.floatValue
                results[index] = value
            }
        }

        return results
    }
}
