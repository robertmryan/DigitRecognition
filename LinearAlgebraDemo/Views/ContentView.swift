//
//  ContentView.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/9/25.
//

import SwiftUI
import os.log

let poi = OSSignposter(subsystem: "ML", category: .pointsOfInterest)

struct ContentView: View {
    @StateObject var viewModel = ViewModel()
    @FocusState private var focusedField: Field?
    @State var updatedImageAndLabel: ImageAndLabel

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
                    HStack(spacing: spacing) {
                        TrainingDataView(
                            title: viewModel.dataType,
                            imageAndLabel: viewModel.imageAndLabel,
                            updatedImageAndLabel: $updatedImageAndLabel
                        )
                        .frame(width: (geometry.size.width - spacing) / 2)

                        ChartView(chartData: viewModel.result)
                            .frame(width: (geometry.size.width - spacing) / 2)
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
            viewModel.testModel(for: updatedImageAndLabel)
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
