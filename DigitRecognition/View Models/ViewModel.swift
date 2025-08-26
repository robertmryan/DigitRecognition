//
//  ViewModel.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/12/25.
//

import Foundation

@MainActor
class ViewModel: ObservableObject {
    @Published var imageAndLabel: ImageAndLabel
    @Published var progress: Float?
    @Published var dataSetSuccess: Float?
    @Published var result: [DataPoint] = []
    @Published var isSuccess: Bool? = true
    @Published var error: (any Error)?
    @Published var imagesAndLabels: [ImageAndLabel]?
    @Published var imagesAndLabelsIndex = 0 { didSet { Task { await testModel(priority: .userInitiated) } } }
    @Published var dataType = "Not Started"
    @Published var modelType: ModelType = .singleLayerPerceptron {
        didSet {
            Task {
                try await updateModel(to: modelType)
            }
        }
    }

    @MachineLearningModelActor
    private var model: (any MachineLearningModel)?

    init() {
        imageAndLabel = ImageAndLabel(imageBytes: Array(repeating: 0, count: 28 * 28), digit: nil) // an empty image
    }

    @MachineLearningModelActor
    func updateModel(to type: ModelType) async throws {
        model = try type.model()
    }

    func loadTests() async {
        do {
            dataType = "Testing Dataset"

            guard
                let imagesUrl = Bundle.main.url(forResource: "t10k-images", withExtension: "idx3-ubyte"),
                let labelsUrl = Bundle.main.url(forResource: "t10k-labels", withExtension: "idx1-ubyte")
            else {
                print("cannot find files")
                return
            }

            var imagesAndLabels: [ImageAndLabel] = []
            progress = 0
            defer { progress = nil }

            let task = Task.detached {
                let sequence = try await IDXSequence(images: imagesUrl, labels: labelsUrl)

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

    var testingTask: Task<Float?, Never>?
    var trainingTask: Task<(), any Error>?

    func testEntireDataSet() async {
        dataSetSuccess = nil
        progress = 0
        defer { progress = nil }

        trainingTask?.cancel()
        testingTask?.cancel()

        let task = Task(priority: .utility) { @MachineLearningModelActor () -> Float? in
            guard let model else { return nil }

            var successCount = 0
            let imagesAndLabels = await imagesAndLabels ?? []
            let totalCount = imagesAndLabels.count
            for imageAndLabel in imagesAndLabels {
                let inference = await model.inference(of: Vector(imageAndLabel.imageBytes.map { Float($0) / 255 }))
                let inferredDigit = model.category(of: inference)
                if inferredDigit == imageAndLabel.digit.flatMap({ Int($0) }) {
                    successCount += 1
                }
                Task { @MainActor [successCount] in
                    self.imageAndLabel = imageAndLabel
                    self.progress = Float(successCount) / Float(totalCount)
                }
            }

            return if totalCount > 0 {
                Float(successCount) / Float(totalCount)
            } else {
                nil
            }
        }
        testingTask = task

        dataSetSuccess = await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }

        imagesAndLabelsIndex = 0
    }

    func testModel(priority: TaskPriority? = nil) async {
        guard
            let imagesAndLabels,
            imagesAndLabelsIndex < imagesAndLabels.count
        else {
            return
        }

        await testModel(for: imagesAndLabels[imagesAndLabelsIndex], priority: priority)
    }

    func testModel(for imageAndLabel: ImageAndLabel, priority: TaskPriority? = nil) async {
        self.imageAndLabel = imageAndLabel
        let imageBytes = imageAndLabel.imageBytes

        let expectedDigit = imageAndLabel.digit.flatMap({ Int($0) })

        let x = Vector(imageBytes.map { Float($0) / 255 })

        let task = Task(priority: priority) { @MachineLearningModelActor () -> (Bool, [DataPoint])? in
            guard let model else {
                return nil
            }

            let y = await model.inference(of: x)
            let predictedDigit = model.category(of: y)
            let dataPoints = (0..<10).map { DataPoint(name: "\($0)", value: y[$0]) }
            return (predictedDigit == expectedDigit, dataPoints)
        }

        if let (isSuccess, dataPoints) = await task.value {
            self.isSuccess = expectedDigit == nil ? nil : isSuccess
            self.result = dataPoints
        }
    }

    func train() async {
        do {
            dataType = "Training Dataset"
            progress = 0
            defer { progress = nil }

            trainingTask?.cancel()
            testingTask?.cancel()

            let task = Task(priority: .userInitiated) { @MachineLearningModelActor [self, modelType] in
                if model == nil {
                    model = try modelType.model()
                }

                let trainingOutputs: [Vector<Float>] = [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]

                guard
                    let imagesUrl = Bundle.main.url(forResource: "train-images", withExtension: "idx3-ubyte"),
                    let labelsUrl = Bundle.main.url(forResource: "train-labels", withExtension: "idx1-ubyte")
                else {
                    print("cannot find files")
                    throw CocoaError(.fileNoSuchFile)
                }

                let sequence = try await IDXSequence(images: imagesUrl, labels: labelsUrl)
                let count = sequence.imagesHeader.count

                var index = 0

                let state = poi.beginInterval(#function)
                defer { poi.endInterval(#function, state) }

                var imagesAndLabels: [ImageAndLabel] = []

                for try await record in sequence {
                    let imageBytes = record.imageBytes.map { Float($0) / 255 }
                    let digit = Int(record.labelBytes.first!)
                    imagesAndLabels.append(ImageAndLabel(imageBytes: record.imageBytes, digit: record.labelBytes.first!))

                    if model!.requiresOnDeviceTraining {
                        let x = Vector(imageBytes)
                        let t = trainingOutputs[digit]
                        model!.train(x: x, t: t)
                    }
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

                await update(imagesAndLabels: imagesAndLabels)
            }
            trainingTask = task

            try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }

            self.imagesAndLabelsIndex = 0
        } catch {
            self.error = error
        }
    }

    func update(imagesAndLabels: sending [ImageAndLabel]) {
        self.imagesAndLabels = imagesAndLabels
        self.imagesAndLabelsIndex = 0
    }
}
