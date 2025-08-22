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
        points.path()
    }
}
