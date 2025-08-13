//
//  TrainingDataView.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/12/25.
//

import SwiftUI

struct TrainingDataView: View {
    let title: String
    let imageAndLabel: ImageAndLabel
    @Binding var updatedImageAndLabel: ImageAndLabel

    let columns = Array(repeating: GridItem(.flexible(), spacing: 1), count: 28)
    private let digits = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

    var body: some View {
        VStack {
            Text(title)
                .font(.title)

            GeometryReader { geometry in
                let spacing: CGFloat = 0
                let lineWidth: CGFloat = 0.5
                let totalSpacing = spacing * CGFloat(28)
                let cellSize = (geometry.size.width - totalSpacing) / 28

                ScrollView {
                    ZStack {
                        LazyVGrid(columns: columns, spacing: spacing) {
                            ForEach(0 ..< imageAndLabel.imageBytes.count, id: \.self) { index in
                                Rectangle()
                                    .fill(color(for: imageAndLabel.imageBytes[index]))
                                    .strokeBorder(Color(.sRGB, white: 0.5, opacity: 1), lineWidth: lineWidth)
                                    .frame(width: cellSize, height: cellSize)
                            }
                        }
                        .border(Color(.sRGB, white: 0.5, opacity: 1), width: lineWidth * 2)

                        DrawingView(imageAndLabel: imageAndLabel, updatedImageAndLabel: $updatedImageAndLabel)
                    }
                }
            }

            Text(imageAndLabel.digit.flatMap { label(for: $0) } ?? "Unknown")
                .font(.title2)
        }
    }

    func color(for byte: UInt8) -> Color {
        Color(.sRGB, white: Double(255-byte) / 255, opacity: 1)
    }

    func label(for byte: UInt8) -> String {
        digits[Int(byte)]
    }
}
