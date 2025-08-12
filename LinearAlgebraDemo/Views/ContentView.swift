//
//  ContentView.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/9/25.
//

import SwiftUI
import os.log
import Charts
import UniformTypeIdentifiers

let poi = OSSignposter(subsystem: "ML", category: .pointsOfInterest)

struct ContentView: View {
    @StateObject var viewModel = ViewModel()

    let columns = Array(repeating: GridItem(.flexible(), spacing: 1), count: 28)

    var body: some View {
        VStack {
            HStack {
                if viewModel.imagesAndLabels != nil {
                    Button {
                        if viewModel.imagesAndLabelsIndex > 0 {
                            viewModel.imagesAndLabelsIndex -= 1
                        }
                    } label: {
                        Image(systemName: "arrowshape.backward.fill")
                    }
                    .disabled(viewModel.imagesAndLabelsIndex == 0)
                }

                VStack {
                    Text(viewModel.dataType)
                        .font(.title)

                    if let imageBytes = viewModel.imageBytes {
                        GeometryReader { geometry in
                            let spacing: CGFloat = 1
                            // total space taken up by gaps in each row
                            let totalSpacing = spacing * CGFloat(28 - 1)
                            // available width after spacing
                            let cellSize = (geometry.size.width - totalSpacing) / 28

                            ScrollView {
                                LazyVGrid(columns: columns, spacing: spacing) {
                                    ForEach(0..<imageBytes.count, id: \.self) { index in
                                        Rectangle()
                                            .fill(viewModel.color(for: imageBytes[index]))
                                            .frame(width: cellSize, height: cellSize)
                                    }
                                }
                            }
                        }
                    }

                    if let digit = viewModel.digit {
                        Text(viewModel.label(for: digit))
                            .font(.title2)
                    }
                }

                VStack {
                    Text("Inference Results")
                        .font(.title)

                    Chart {
                        ForEach(viewModel.result, id: \.self) { result in
                            BarMark(
                                x: .value("Name", result.name),
                                y: .value("Score", result.value)
                            )
                        }
                    }
                    .chartYScale(domain: 0.0 ... 1.0)
                    .chartYAxis {
                        AxisMarks(values: Array(stride(from: 0.0, through: 1.0, by: 0.1))) { value in
                            AxisGridLine()
                            AxisTick()
                            AxisValueLabel {   // trailing alignment
                                if let doubleValue = value.as(Double.self) {
                                    Text(doubleValue, format: .percent)
                                        .frame(maxWidth: .infinity, alignment: .trailing)
                                }
                            }
                        }
                    }

                }
                if viewModel.imagesAndLabels != nil {
                    Button {
                        viewModel.imagesAndLabelsIndex += 1
                    } label: {
                        Image(systemName: "arrowshape.forward.fill")
                    }
                }
            }

            Button("Train Model") {
                Task {
                    await viewModel.train()
                }
            }

            Button("Test Model") {
                Task {
                    await viewModel.loadTests()
                }
            }

            if let progress = viewModel.progress {
                ProgressView(value: progress)
            }
        }
        .padding()
    }

}

struct DataPoint: Hashable {
    let name: String
    let value: Float
}

struct ImageAndLabel: Hashable {
    let imageBytes: [UInt8]
    let digit: UInt8
}

@MainActor
class ViewModel: ObservableObject {
    @Published var imageBytes: [UInt8]?
    @Published var digit: UInt8?
    @Published var progress: Float?
    @Published var result: [DataPoint] = []
    @Published var error: (any Error)?
    @Published var imagesAndLabels: [ImageAndLabel]?
    @Published var imagesAndLabelsIndex = 0 { didSet { testModel() } }
    @Published var dataType = "Not Started"

    var model: Matrix<Float>?
    var biasVector: Vector<Float>?

    private let digits = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    private let outputs: [Vector<Float>] = [
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
                            self.imageBytes = record.imageBytes
                            self.digit = record.labelBytes.first
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

        let imageBytes = imagesAndLabels[imagesAndLabelsIndex].imageBytes
        self.imageBytes = imageBytes
        let digit = imagesAndLabels[imagesAndLabelsIndex].digit
        self.digit = digit

        let x = Vector(imageBytes.map { Float($0) / 255 })
        guard let model, let biasVector else {
            return
        }

        let y = LinearAlgebra.test(x: x, w: model, b: biasVector)
        self.result = (0..<10).map { DataPoint(name: "\($0)", value: y[$0]) }
    }

    func train() async {
        dataType = "Training Data"
        do {
            guard
                let imagesUrl = Bundle.main.url(forResource: "train-images", withExtension: "idx3-ubyte"),
                let labelsUrl = Bundle.main.url(forResource: "train-labels", withExtension: "idx1-ubyte")
            else {
                print("cannot find files")
                return
            }

            let sequence = try await IDXSequence(images: imagesUrl, labels: labelsUrl)
            let count = sequence.imagesHeader.count
            let modelColumns = sequence.imagesHeader.countPerItem // 784 for the 28 × 28 image
            let modelRows = 10                                    // 10
            let learningRate: Float = 0.01

            let model = Matrix<Float>(repeating: 0.01, rows: modelRows, cols: modelColumns)
            let biasVector = Vector<Float>(repeating: 0, count: modelRows)

            progress = 0
            defer { progress = nil }

            var index = 0

            let task = Task.detached { [outputs] in
                let state = poi.beginInterval(#function)
                defer { poi.endInterval(#function, state) }

                var imagesAndLabels: [ImageAndLabel] = []

                for try await record in sequence {
                    let imageBytes = record.imageBytes.map { Float($0) / 255 }
                    let digit = Int(record.labelBytes.first!)
                    imagesAndLabels.append(ImageAndLabel(imageBytes: record.imageBytes, digit: record.labelBytes.first!))

                    let x = Vector(imageBytes)
                    let t = outputs[digit]
                    LinearAlgebra.trainStep(
                        x: x,
                        t: t,
                        w: model,
                        b: biasVector,
                        learningRate: learningRate
                    )

                    let progress = Float(index) / Float(count)

                    try Task.checkCancellation()
                    Task { @MainActor in
                        self.imageBytes = record.imageBytes
                        self.digit = record.labelBytes.first
                        self.progress = progress
                    }

                    index += 1

                    // if index < 100 {
                    //     self.createImage(index: index, from: imageBytes)
                    // }
                }
                return imagesAndLabels
            }

            try await withTaskCancellationHandler {
                imagesAndLabels = try await task.value
            } onCancel: {
                task.cancel()
            }

            self.model = model
            self.biasVector = biasVector
            self.imagesAndLabelsIndex = 0
        } catch {
            self.error = error
        }
    }

    nonisolated func createImage(index: Int, from imageData: [Float]) {
        #if os(iOS)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard
            let context = CGContext(
                data: nil,
                width: 28,
                height: 28,
                bitsPerComponent: 8,
                bytesPerRow: 28 * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue
            ),
            let buffer = context.data
        else {
            return
        }
        let pixelBuffer = buffer.bindMemory(to: UInt8.self, capacity: 28 * 4 * 28 * 4)
        for (index, pixel) in imageData.enumerated() {
            print(UInt8(255 * (1 - pixel)))
            let offset = index * 4
            pixelBuffer[offset] = UInt8(255 * (1 - pixel))
            pixelBuffer[offset + 1] = UInt8(255 * (1 - pixel))
            pixelBuffer[offset + 2] = UInt8(255 * (1 - pixel))
            pixelBuffer[offset + 3] = 255
        }

        let cgImage = context.makeImage()!
        let image = UIImage(cgImage: cgImage)
        guard let data = image.pngData() else {
            return
        }

        let folder = URL.temporaryDirectory.appending(path: "MNIST images")
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        print(folder)
        let fileUrl = folder.appendingPathComponent("\(index).png")
        try? FileManager.default.removeItem(at: fileUrl)
        try? data.write(to: fileUrl)


//        let mutableData = CFDataCreateMutable(nil, 0)!
//        guard let destination = CGImageDestinationCreateWithData(mutableData, UTType.jpeg.identifier as CFString, 1, nil) else {
//            return
//        }
//
//        let options = [
//            kCGImageDestinationLossyCompressionQuality: 0.9
//        ] as CFDictionary
//
//        CGImageDestinationAddImage(destination, cgImage, options)
//        guard CGImageDestinationFinalize(destination) else {
//            return
//        }
//
//        let data = mutableData as Data
//
//        let folder = URL.temporaryDirectory.appending(path: "MNIST images")
//        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
//        print(folder)
//        let fileUrl = folder.appendingPathComponent("\(index).jpg")
//        try? FileManager.default.removeItem(at: fileUrl)
//        try? data.write(to: fileUrl)

//        let nsImage = NSImage(cgImage: cgImage, size: CGSize(width: 28, height: 28))
//        if let data = nsImage.tiffRepresentation {
//            let folder = URL.temporaryDirectory.appending(path: "MNIST images")
//            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
//            print(folder)
//            let fileUrl = folder.appendingPathComponent("\(index).tiff")
//            try? FileManager.default.removeItem(at: fileUrl)
//            try? data.write(to: fileUrl)
//        }
        #endif
    }

    func color(for byte: UInt8) -> Color {
        Color(white: Double(255-byte) / 255)
    }

    func label(for byte: UInt8) -> String {
        digits[Int(byte)]
    }
}

#Preview {
    ContentView()
}
