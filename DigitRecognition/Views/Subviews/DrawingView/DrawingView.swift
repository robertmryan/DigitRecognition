//
//  DrawingView.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/12/25.
//

import SwiftUI

// MARK: - Drawing Pad
struct DrawingView: View {
    @Binding var strokes: [Stroke]
    @Binding var currentPoints: [CGPoint]

    // Tweak these as you like
    let lineWidth: CGFloat
    let lineColor: Color = .blue

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Render all completed strokes + the one in progress
                Canvas { context, _ in
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
}
