//
//  ModelPicker.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/24/25.
//

import SwiftUI

struct ModelPicker: View {
    @Binding var modelType: ModelType

    var body: some View {
        Picker(selection: $modelType) {
            ForEach(ModelType.allCases) { model in
                Text(model.label).tag(model)
            }
        } label: {
            Text("Model:")
        }
        .pickerStyle(.menu)
        .fixedSize()
    }
}
