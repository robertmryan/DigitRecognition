//
//  ModelView.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/27/25.
//

import SwiftUI

struct ModelView: View {
    @Binding var modelType: ModelType
    @State var shownImage = 0

    var imageNames: [String] {
        [modelType.imageName, "all_digits", "just_4"]
    }

    var body: some View {
        VStack {
            Text(modelType.longLabel)
                .font(.title)
                .padding()

            Image(imageNames[shownImage])
                .resizable()
                .scaledToFit()
                .padding()
                .onTapGesture {
                    shownImage = (shownImage + 1) % imageNames.count
                }
        }
    }
}

#Preview {
    ModelView(modelType: .constant(.singleLayerPerceptron))
}
