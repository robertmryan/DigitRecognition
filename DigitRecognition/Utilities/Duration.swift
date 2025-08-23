//
//  Duration.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/23/25.
//

import Foundation

extension Duration {
    var seconds: Float {
        let (seconds, attoseconds) = components
        return Float(seconds) + Float(attoseconds) * 1e-18
    }
}
