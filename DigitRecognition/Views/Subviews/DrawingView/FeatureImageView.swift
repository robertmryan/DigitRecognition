//
//  FeatureImageView.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/23/25.
//

import SwiftUI

struct FeatureImageView: View {
    let imageAndLabel: ImageAndLabel

    var body: some View {
        if let cgImage = imageAndLabel.image() {
            Image(decorative: cgImage, scale: 1)
        }
    }
}

 #Preview {
     FeatureImageView(imageAndLabel: ImageAndLabel.example)
 }
