//
//  ViewModel.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/12/25.
//

import Foundation

@MainActor
class ViewModel: ObservableObject {
    @Published var imageAndLabel: ImageAndLabel
    @Published var progress: Float?
    @Published var result: [DataPoint] = []
    @Published var error: (any Error)?
    @Published var imagesAndLabels: [ImageAndLabel]?
    @Published var imagesAndLabelsIndex = 0 { didSet { testModel() } }
    @Published var dataType = "Not Started"

    private var model: (any MachineLearningModel)?

    private let trainingOutputs: [Vector<Float>] = [
        Vector([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        Vector([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        Vector([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        Vector([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        Vector([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        Vector([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        Vector([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        Vector([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        Vector([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        Vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    ]

    init() {
        imageAndLabel = ImageAndLabel(imageBytes: Array(repeating: 0, count: 28 * 28), digit: nil) // an empty image
    }

    func loadTests() async {
        do {
            dataType = "Testing Data"

            guard
                let imagesUrl = Bundle.main.url(forResource: "t10k-images", withExtension: "idx3-ubyte"),
                let labelsUrl = Bundle.main.url(forResource: "t10k-labels", withExtension: "idx1-ubyte")
            else {
                print("cannot find files")
                return
            }

            let sequence = try await IDXSequence(images: imagesUrl, labels: labelsUrl)
            var imagesAndLabels: [ImageAndLabel] = []
            progress = 0
            defer { progress = nil }

            let task = Task.detached {
                let count = sequence.imagesHeader.count
                var index = 0

                for try await record in sequence {
                    try Task.checkCancellation()
                    let imageBytes = record.imageBytes
                    if let digit = record.labelBytes.first {
                        imagesAndLabels.append(ImageAndLabel(imageBytes: imageBytes, digit: digit))

                        index += 1
                        let progress = Float(index) / Float(count)

                        Task { @MainActor in
                            self.imageAndLabel = ImageAndLabel(imageBytes: record.imageBytes, digit: record.labelBytes.first)
                            self.progress = progress
                        }
                    }
                }
                return imagesAndLabels
            }
            try await withTaskCancellationHandler {
                self.imagesAndLabels = try await task.value
            } onCancel: {
                task.cancel()
            }

            self.imagesAndLabelsIndex = 0
        } catch {
            self.error = error
        }
    }

    func testModel() {
        guard
            let imagesAndLabels,
            imagesAndLabelsIndex < imagesAndLabels.count
        else {
            return
        }

        testModel(for: imagesAndLabels[imagesAndLabelsIndex])
    }

    func testModel(for imageAndLabel: ImageAndLabel) {
        self.imageAndLabel = imageAndLabel
        let imageBytes = imageAndLabel.imageBytes

        let x = Vector(imageBytes.map { Float($0) / 255 })
        guard let model else {
            return
        }

        let y = model.inference(of: x)
        self.result = (0..<10).map { DataPoint(name: "\($0)", value: y[$0]) }
    }

    func train() async {
        dataType = "Training Data"
        do {
            progress = 0
            defer { progress = nil }

            let task = Task.detached(priority: .utility) { [self, trainingOutputs] in
                guard
                    let imagesUrl = Bundle.main.url(forResource: "train-images", withExtension: "idx3-ubyte"),
                    let labelsUrl = Bundle.main.url(forResource: "train-labels", withExtension: "idx1-ubyte")
                else {
                    print("cannot find files")
                    throw CocoaError(.fileNoSuchFile)
                }

                let sequence = try await IDXSequence(images: imagesUrl, labels: labelsUrl)
                let count = sequence.imagesHeader.count
                let modelColumns = sequence.imagesHeader.countPerItem // 784 for the 28 × 28 image
                let modelRows = 10                                    // 10

                let model = SGDSingleLayer(inputVectorSize: modelRows, outputVectorSize: modelColumns)

                var index = 0

                let state = poi.beginInterval(#function)
                defer { poi.endInterval(#function, state) }

                var imagesAndLabels: [ImageAndLabel] = []

                for try await record in sequence {
                    let imageBytes = record.imageBytes.map { Float($0) / 255 }
                    let digit = Int(record.labelBytes.first!)
                    imagesAndLabels.append(ImageAndLabel(imageBytes: record.imageBytes, digit: record.labelBytes.first!))

                    let x = Vector(imageBytes)
                    let t = trainingOutputs[digit]
                    model.train(x: x, t: t)

                    let progress = Float(index) / Float(count)

                    try Task.checkCancellation()
                    Task { @MainActor in
                        self.imageAndLabel = ImageAndLabel(imageBytes: record.imageBytes, digit: record.labelBytes.first)
                        self.progress = progress
                    }

                    index += 1

                    // if index < 100 {
                    //     self.createImage(index: index, from: imageBytes)
                    // }
                }

                await update(model: model, imagesAndLabels: imagesAndLabels)
            }

            try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
        } catch {
            self.error = error
        }
    }

    func update(model: any MachineLearningModel, imagesAndLabels: [ImageAndLabel]) {
        self.model = model
        self.imagesAndLabels = imagesAndLabels
        self.imagesAndLabelsIndex = 0
    }
}
