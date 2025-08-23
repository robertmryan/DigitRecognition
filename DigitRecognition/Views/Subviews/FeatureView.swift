//
//  FeatureView.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/12/25.
//

import SwiftUI

private let featureDimension = 28
private let strokeLineWidth: CGFloat = 2

struct FeatureView: View {
    let title: String
    let imageAndLabel: ImageAndLabel
    @Binding var updatedImageAndLabel: ImageAndLabel
    @State private var strokes: [Stroke] = []
    @State private var currentPoints: [CGPoint] = []

    let columns = Array(repeating: GridItem(.flexible(), spacing: 1), count: featureDimension)
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
                    if let imageAndLabel = rasterizeStrokesToImageAndLabel(strokes: strokes, lineWidth: strokeLineWidth) {
                        updatedImageAndLabel = imageAndLabel
                    }
                    strokes.removeAll()
                    currentPoints.removeAll()
                }
                .disabled(strokes.isEmpty)

                Spacer()

                FeatureImageView(imageAndLabel: imageAndLabel)
            }

            GeometryReader { geometry in
                let spacing: CGFloat = 1
                let lineWidth: CGFloat = 0
                let totalSpacing = spacing * CGFloat(featureDimension - 1)
                let cellSize = CGFloat(Int((min(geometry.size.width, geometry.size.height) - totalSpacing) / CGFloat(featureDimension)))
                let totalSize = cellSize * CGFloat(featureDimension) + totalSpacing

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
                    .frame(width: totalSize, height: totalSize)

                    DrawingView(
                        strokes: $strokes,
                        currentPoints: $currentPoints,
                        lineWidth: strokeLineWidth * geometry.size.width / CGFloat(featureDimension)
                    )
                }
            }

            Text(imageAndLabel.digit.flatMap { label(for: $0) } ?? "Unknown")
                .font(.title2)
        }
    }
}

// MARK: - Private utilities for displaying model parameters

private extension FeatureView {
    func color(for byte: UInt8) -> Color {
        Color(.sRGB, white: Double(255-byte) / 255, opacity: 1)
    }

    func label(for byte: UInt8) -> String {
        digits[Int(byte)]
    }
}

// MARK: - Private utilities for converting strokes to pixel data

private extension FeatureView {
    func rasterizeStrokesToImageAndLabel(
        strokes: [Stroke],
        imageSize: CGSize = CGSize(width: 28, height: 28),
        lineWidth: CGFloat = 2
    ) -> ImageAndLabel? {
        let translation = translationOfGeometricCenter(lineWidth: strokeLineWidth)

        guard let bytes = rasterizeStrokesToMNISTPixelBytes(strokes: strokes, imageSize: imageSize, lineWidth: lineWidth, translation: translation) else {
            return nil
        }
        return ImageAndLabel(imageBytes: bytes, digit: nil)
    }

    func translationOfGeometricCenter(lineWidth: CGFloat = 2) -> CGAffineTransform {
        let width = 300
        let height = 300
        guard let bytes = rasterizeStrokesToMNISTPixelBytes(
            strokes: strokes,
            imageSize: CGSize(width: width, height: height),
            lineWidth: lineWidth * CGFloat(width) / CGFloat(featureDimension)
        ) else { return .identity }

        var total = 0
        var sumX  = 0
        var sumY  = 0

        for x in 0 ..< width {
            for y in 0 ..< height {
                let w = Int(bytes[y * width + x])          // weight (use grayscale intensity; or 1 if binarized)
                total += w
                sumX += x * w
                sumY += y * w
            }
        }

        if total == 0 {
            print("total == 0")
            return .identity
        }

        let originalPixelsX: CGFloat = CGFloat(sumX) / CGFloat(total) - CGFloat(width) / 2
        let originalPixelsY: CGFloat = CGFloat(sumY) / CGFloat(total) - CGFloat(width) / 2
        let pixelsX: CGFloat = -14 * originalPixelsX / CGFloat(width)
        let pixelsY: CGFloat = 14 * originalPixelsY / CGFloat(height)

        return CGAffineTransform(
            translationX: pixelsX,
            y: pixelsY
        )
    }

    func rasterizeStrokesToMNISTPixelBytes(
        strokes: [Stroke],
        imageSize: CGSize = CGSize(width: 28, height: 28),
        lineWidth: CGFloat = 2,
        translation: CGAffineTransform = .identity
    ) -> [UInt8]? {
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
        else { return nil }

        let path = path(for: strokes)
        let cgPath = path.cgPath

        let transform = aspectFitTransform(for: cgPath, imageRect: CGRect(origin: .zero, size: imageSize), imageStrokeWidth: lineWidth)
            .concatenating(translation)

        context.saveGState(); defer { context.restoreGState() }
        context.concatenate(transform)

        // Scale strokes from your canvas coordinate space to 28×28
        // Assume you know the drawing bounds, e.g., 300×300 canvas:

        context.setFillColor(gray: 0, alpha: 1) // background black
        context.fill(CGRect(origin: .zero, size: imageSize))

        context.setStrokeColor(gray: 1, alpha: 1) // white strokes
        context.setLineCap(.round)
        context.setLineJoin(.round)
        context.setLineWidth(lineWidth / transform.scale.x)

        context.addPath(cgPath)
        context.strokePath()

        let bufferPointer = ptr.assumingMemoryBound(to: UInt8.self)

        let buffer = UnsafeMutableBufferPointer(
            start: bufferPointer,
            count: width * height
        )

        let bytes = Array(buffer)

        for index in 0 ..< buffer.count {
            buffer[index] = 255 - buffer[index]
        }

        return bytes
    }

    func aspectFitTransform(
        for path: CGPath,
        imageRect: CGRect = CGRect(x: 0, y: 0, width: 28, height: 28),
        inset: CGFloat = 4,
        imageStrokeWidth: CGFloat
    ) -> CGAffineTransform {
        let src = path.boundingBoxOfPath
        let dest = imageRect.insetBy(dx: inset + imageStrokeWidth / 2, dy: inset + imageStrokeWidth / 2)

        guard src.width > 0, src.height > 0, dest.width > 0, dest.height > 0 else { return .identity }

        let s  = min(dest.width / src.width, dest.height / src.height) // uniform
        let scaled = CGSize(width: src.width * s, height: src.height * s)

        // Center inside the *inset* box.
        let translationX = dest.midX - (src.minX * s) - scaled.width  * 0.5
        let translationY = dest.midY - (src.minY * s) - scaled.height * 0.5

        // Flip Y inside the full image, then place the fitted path.
        return CGAffineTransform(translationX: 0, y: imageRect.maxY)
            .scaledBy(x: 1, y: -1)
            .translatedBy(x: translationX, y: translationY)
            .scaledBy(x: s, y: s)
    }

    func path(for strokes: [Stroke]) -> Path {
        var path = Path()

        for stroke in strokes {
            path.addPath(stroke.points.path())
        }

        return path
    }
}
