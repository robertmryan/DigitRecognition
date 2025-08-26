//
//  ContentView.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/9/25.
//

import SwiftUI
import os.log

let poi = OSSignposter(subsystem: "SGDSingleLayer", category: .pointsOfInterest)

struct ContentView: View {
    @StateObject var viewModel = ViewModel()
    @FocusState private var focusedField: Field?
    @State var updatedImageAndLabel: ImageAndLabel
    @State var elapsed: Duration?

    init() {
        updatedImageAndLabel = ImageAndLabel(imageBytes: Array(repeating: 0, count: 28 * 28), digit: nil)
    }

    let columns = Array(repeating: GridItem(.flexible(), spacing: 1), count: 28)
    let spacing: CGFloat = 20

    var body: some View {
        VStack {
            HStack {
                if viewModel.imagesAndLabels != nil {
                    Button {
                        previous()
                    } label: {
                        Image(systemName: "arrowshape.backward.fill")
                    }
                    .disabled(viewModel.imagesAndLabelsIndex == 0)
                }

                GeometryReader { geometry in
                    let size = sizeOfFeatureView(in: geometry.size)

                    AdaptiveStack(spacing: spacing) {
                        FeatureView(
                            imageAndLabel: $viewModel.imageAndLabel,
                            updatedImageAndLabel: $updatedImageAndLabel
                        )
                        .frame(width: size.width, height: size.height)

                        VStack {
                            PredictionView(
                                chartData: $viewModel.result,
                                isSuccess: $viewModel.isSuccess,
                                progress: $viewModel.progress,
                                elapsed: $elapsed,
                                imagesAndLabels: $viewModel.imagesAndLabels
                            )

                            if let dataSetSuccess = viewModel.dataSetSuccess {
                                Text("Total accuracy: \(dataSetSuccess, format: .percent.precision(.fractionLength(1))) in \(viewModel.dataType)")
                            }
                        }
                    }
                }

                if viewModel.imagesAndLabels != nil {
                    Button {
                        next()
                    } label: {
                        Image(systemName: "arrowshape.forward.fill")
                    }
                    .disabled(viewModel.imagesAndLabelsIndex >= viewModel.imagesAndLabels!.count - 1)
                }
            }

            HStack {
                Spacer()

                ModelPicker(modelType: $viewModel.modelType)

                Spacer()

                Button("Training") {
                    Task {
                        let start = ContinuousClock.now
                        await viewModel.train()
                        await viewModel.testEntireDataSet()
                        elapsed = ContinuousClock.now - start
                    }
                }
                .disabled(viewModel.modelType == .convolutionalNeuralNetwork)

                Spacer()

                Button("Testing") {
                    Task {
                        let start = ContinuousClock.now
                        await viewModel.loadTests()
                        await viewModel.testEntireDataSet()
                        elapsed = ContinuousClock.now - start
                    }
                }

                Spacer()
            }
        }
        .padding()
        .focusable()
        .focusEffectDisabled()
        .focused($focusedField, equals: .main)
        .onKeyPress { key in
            switch key.key {
                case .leftArrow:  previous()
                case .rightArrow: next()
                default:          return .ignored
            }

            return .handled
        }
        .onAppear {
            focusedField = .main
        }
        .task(id: updatedImageAndLabel) {
            await viewModel.testModel(for: updatedImageAndLabel)
        }
    }

    func sizeOfFeatureView(in size: CGSize) -> CGSize {
        if size.width > size.height {
            return CGSize(width: min(size.height * 0.9, size.width * 2 / 3), height: size.height)
        } else {
            return CGSize(width: size.width, height: min((size.width / 1) + 90, size.height * 2 / 3))
        }
    }

    func previous() {
        guard viewModel.imagesAndLabelsIndex > 0 else { return }

        Task {
            viewModel.imagesAndLabelsIndex -= 1
        }
    }

    func next() {
        guard viewModel.imagesAndLabelsIndex < viewModel.imagesAndLabels!.count - 1 else { return }
        Task {
            viewModel.imagesAndLabelsIndex += 1
        }
    }
}

enum Field {
    case main
}

#Preview {
    ContentView()
}
