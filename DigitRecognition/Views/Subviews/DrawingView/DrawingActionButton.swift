//
//  DrawingActionButton.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/21/25.
//

import SwiftUI

struct DrawingActionButton: View {
    let text: String
    let action: () -> Void

    init(_ text: String, action: @escaping () -> Void) {
        self.text = text
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Text(text)
                .padding(.horizontal, 24)
                .padding(.vertical, 12)
                .background(Color(.sRGB, white: 0.25, opacity: 1))
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)    }
}

#Preview {
    DrawingActionButton("Sample") { }
}
