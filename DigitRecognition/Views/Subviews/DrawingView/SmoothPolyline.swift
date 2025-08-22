//
//  SmoothPolyline.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/12/25.
//

import SwiftUI

// MARK: - Smooth polyline shape (quad-curve smoothing)
struct SmoothPolyline: Shape {
    var points: [CGPoint]

    func path(in rect: CGRect) -> Path {
        var path = Path()
        guard points.count > 1 else {
            if let p = points.first { path.addEllipse(in: CGRect(origin: p, size: .zero)) }
            return path
        }

        path.move(to: points[0])

        // Draw quadratic curves between midpoints for a smoother line
        for i in 1..<points.count {
            let mid = CGPoint(
                x: (points[i].x + points[i - 1].x) / 2,
                y: (points[i].y + points[i - 1].y) / 2
            )
            path.addQuadCurve(to: mid, control: points[i - 1])
        }

        // Finish the last segment
        if let last = points.last { path.addLine(to: last) }
        return path
    }
}
