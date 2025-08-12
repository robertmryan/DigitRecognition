//
//  DrawingView.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/12/25.
//

import SwiftUI

// MARK: - Drawing Pad
struct DrawingView: View {
    @State private var strokes: [Stroke] = []
    @State private var currentPoints: [CGPoint] = []
    let imageAndLabel: ImageAndLabel
    @Binding var updatedImageAndLabel: ImageAndLabel

    // Tweak these as you like
    let lineWidth: CGFloat = 36
    let lineColor: Color = .blue

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Render all completed strokes + the one in progress
                Canvas {
                    context,
                    _ in
                    // Completed strokes
                    for stroke in strokes {
                        let shape = SmoothPolyline(points: stroke.points)
                        let path = shape.path(in: .infinite)
                        context.stroke(
                            path,
                            with: .color(stroke.color),
                            style: StrokeStyle(
                                lineWidth: stroke.lineWidth,
                                lineCap: .round,
                                lineJoin: .round
                            )
                        )
                    }

                    // Current in-progress stroke
                    if !currentPoints.isEmpty {
                        let shape = SmoothPolyline(points: currentPoints)
                        let path = shape.path(in: .infinite)
                        context.stroke(
                            path,
                            with: .color(lineColor),
                            style: StrokeStyle(
                                lineWidth: lineWidth,
                                lineCap: .round,
                                lineJoin: .round
                            )
                        )
                    }
                }
                .drawingGroup() // offscreen render for smoother curves

                // Simple controls (optional)
                VStack {
                    HStack {
                        Button("Undo") {
                            _ = strokes.popLast()
                        }
                        .background(Color(.sRGB, white: 0.25, opacity: 0.8))
                        .disabled(strokes.isEmpty)

                        Button("Clear") {
                            strokes.removeAll()
                            currentPoints.removeAll()
                        }
                        .background(Color(.sRGB, white: 0.25, opacity: 0.8))

                        Button("Process") {
                            rasterizeStrokesToMNISTPixels(strokes: strokes, lineWidth: lineWidth, originalSize: geometry.size)
                            strokes.removeAll()
                            currentPoints.removeAll()
                        }
                        .background(Color(.sRGB, white: 0.25, opacity: 0.8))

                        Spacer()
                    }
                    .padding()
                    Spacer()
                }
            }
            .contentShape(Rectangle()) // make entire area hit-testable
            .gesture(
                DragGesture(minimumDistance: 0, coordinateSpace: .local)
                    .onChanged { value in
                        // Append the current location; you can also throttle for perf if needed
                        currentPoints.append(value.location)
                    }
                    .onEnded { _ in
                        guard currentPoints.count > 1 else {
                            currentPoints.removeAll()
                            return
                        }
                        strokes.append(Stroke(points: currentPoints,
                                              color: lineColor,
                                              lineWidth: lineWidth))
                        currentPoints.removeAll()
                    }
            )
        }
        .background(Color.clear)
    }

    private func rasterizeStrokesToMNISTPixels(
        strokes: [Stroke],
        imageSize: CGSize = CGSize(width: 28, height: 28),
        lineWidth: CGFloat = 2,
        originalSize: CGSize
    ) {
        // 1. Create a Core Graphics context
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bytesPerPixel = 1
        let bytesPerRow = Int(imageSize.width) * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: Int(imageSize.width * imageSize.height))

        pixelData.withUnsafeMutableBytes { ptr in
            if let context = CGContext(
                data: ptr.baseAddress,
                width: Int(imageSize.width),
                height: Int(imageSize.height),
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) {
                // Scale strokes from your canvas coordinate space to 28×28
                // Assume you know the drawing bounds, e.g., 300×300 canvas:
                let originalSize: CGFloat = originalSize.width // your actual drawing area
                let scale = imageSize.width / originalSize

                context.setFillColor(gray: 0, alpha: 1) // background black
                context.fill(CGRect(origin: .zero, size: imageSize))

                context.setStrokeColor(gray: 1, alpha: 1) // white strokes
                context.setLineCap(.round)
                context.setLineJoin(.round)
                context.setLineWidth(lineWidth * scale)

                for stroke in strokes {
                    guard let first = stroke.points.first else { continue }
                    context.beginPath()
                    context.move(
                        to: CGPoint(
                            x: first.x * scale,
                            y: (originalSize - first.y) * scale
                        )
                    ) // flip Y for CG
                    for p in stroke.points.dropFirst() {
                        context.addLine(
                            to: CGPoint(
                                x: p.x * scale,
                                y: (originalSize - p.y) * scale
                            )
                        )
                    }
                    context.strokePath()
                }
            }
        }

        self.updatedImageAndLabel = ImageAndLabel(imageBytes: pixelData, digit: nil)
    }
}

// MARK: - Stroke model
private struct Stroke: Identifiable, Hashable {
    let id = UUID()
    var points: [CGPoint]
    var color: Color = .primary
    var lineWidth: CGFloat = 6
}
