//
//  TrainingDataView.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/12/25.
//

import SwiftUI

struct TrainingDataView: View {
    let title: String
    let imageAndLabel: ImageAndLabel
    @Binding var updatedImageAndLabel: ImageAndLabel
    @State private var strokes: [Stroke] = []
    @State private var currentPoints: [CGPoint] = []
    let strokeLineWidth: CGFloat = 36

    let columns = Array(repeating: GridItem(.flexible(), spacing: 1), count: 28)
    private let digits = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

    var body: some View {
        VStack {
            Text(title)
                .font(.title)

            HStack {
                DrawingActionButton("Undo") {
                    _ = strokes.popLast()
                }
                .disabled(strokes.isEmpty)

                DrawingActionButton("Clear") {
                    strokes.removeAll()
                    currentPoints.removeAll()
                }
                .disabled(strokes.isEmpty)

                DrawingActionButton("Process") {
                    rasterizeStrokesToMNISTPixels(strokes: strokes, lineWidth: strokeLineWidth)
                    strokes.removeAll()
                    currentPoints.removeAll()
                }
                .disabled(strokes.isEmpty)

                Spacer()
            }
            .padding()

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

                        DrawingView(strokes: $strokes, currentPoints: $currentPoints, lineWidth: strokeLineWidth)
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

    private func rasterizeStrokesToMNISTPixels(
        strokes: [Stroke],
        imageSize: CGSize = CGSize(width: 28, height: 28),
        lineWidth: CGFloat = 2
    ) {
        // 1. Create a Core Graphics context
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bytesPerPixel = 1
        let bytesPerRow = Int(imageSize.width) * bytesPerPixel
        let width = Int(imageSize.width)
        let height = Int(imageSize.height)

        guard
            let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ),
            let ptr = context.data
        else { return }

        let path = path(for: strokes, lineWidth: lineWidth)
        let cgPath = path.cgPath

        let t = aspectFitTransform(for: cgPath)

        context.saveGState(); defer { context.restoreGState() }
        context.concatenate(t)

        // Scale strokes from your canvas coordinate space to 28×28
        // Assume you know the drawing bounds, e.g., 300×300 canvas:

        context.setFillColor(gray: 0, alpha: 1) // background black
        context.fill(CGRect(origin: .zero, size: imageSize))

        context.setStrokeColor(gray: 1, alpha: 1) // white strokes
        context.setLineCap(.round)
        context.setLineJoin(.round)
        context.setLineWidth(36)

        context.addPath(cgPath)
        context.strokePath()

        let buffer = UnsafeBufferPointer<UInt8>(
            start: ptr.assumingMemoryBound(to: UInt8.self),
            count: width * height
        )
        let bytes = Array(buffer)

        self.updatedImageAndLabel = ImageAndLabel(imageBytes: bytes, digit: nil)
    }

    func aspectFitTransform(
        for path: CGPath,
        imageRect: CGRect = CGRect(x: 0, y: 0, width: 28, height: 28),
        inset: CGFloat = 4
    ) -> CGAffineTransform {
        let dst = imageRect.insetBy(dx: inset, dy: inset)   // e.g. (4,4,20,20)
        let src = path.boundingBoxOfPath
        guard src.width > 0, src.height > 0, dst.width > 0, dst.height > 0 else { return .identity }

        let s  = min(dst.width / src.width, dst.height / src.height) // uniform
        let scaled = CGSize(width: src.width * s, height: src.height * s)

        // Center inside the *inset* box.
        let tx = dst.midX - (src.minX * s) - scaled.width  * 0.5
        let ty = dst.midY - (src.minY * s) - scaled.height * 0.5

        // Flip Y inside the full image, then place the fitted path.
        return CGAffineTransform(translationX: 0, y: imageRect.maxY)
            .scaledBy(x: 1, y: -1)
            .translatedBy(x: tx, y: ty)
            .scaledBy(x: s, y: s)
    }

    func path(for strokes: [Stroke], lineWidth: CGFloat) -> Path {
        var path = Path()

        for stroke in strokes {
            guard let firstPoint = stroke.points.first else { continue }
            path.move(to: firstPoint)

            for point in stroke.points.dropFirst() {
                path.addLine(to: point)
            }
        }

        return path
    }
}
