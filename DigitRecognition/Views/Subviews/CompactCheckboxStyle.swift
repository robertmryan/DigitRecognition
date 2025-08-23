//
//  CompactCheckboxStyle.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/23/25.
//

import SwiftUI

struct CompactCheckboxStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        Button {
            configuration.isOn.toggle()
        } label: {
            Image(systemName: configuration.isOn ? "checkmark.square.fill" : "square")
                .imageScale(.large)
                .padding(6)            // small hit target bump
                .contentShape(Rectangle())
                .accessibilityLabel(configuration.isOn ? "On" : "Off")
        }
        .buttonStyle(.plain)
    }
}
