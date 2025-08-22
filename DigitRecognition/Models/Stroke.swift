//
//  Stroke.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/21/25.
//

import SwiftUI

struct Stroke: Identifiable, Hashable {
    let id = UUID()
    var points: [CGPoint]
    var color: Color = .primary
    var lineWidth: CGFloat = 6
}
