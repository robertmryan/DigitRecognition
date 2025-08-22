//
//  ImageAndLabel.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/12/25.
//

import Foundation

/// An image and its associated label.

struct ImageAndLabel: Hashable {
    let imageBytes: [UInt8]
    let digit: UInt8?
}
