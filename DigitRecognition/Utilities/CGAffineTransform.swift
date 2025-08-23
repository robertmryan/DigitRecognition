//
//  CGAffineTransform.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/22/25.
//

import Foundation

extension CGAffineTransform {
    /// Returns the uniform scale factors (x and y) implied by the transform.
    var scale: (x: CGFloat, y: CGFloat) {
        // The first column (a, c) gives the transformed x-axis.
        let scaleX = sqrt(a * a + c * c)
        // The second column (b, d) gives the transformed y-axis.
        let scaleY = sqrt(b * b + d * d)
        return (x: scaleX, y: scaleY)
    }
}
