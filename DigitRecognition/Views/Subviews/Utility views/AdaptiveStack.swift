//
//  AdaptiveStack.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/26/25.
//

import SwiftUI

struct AdaptiveStack<Content: View>: View {
    let spacing: CGFloat?
    @ViewBuilder var content: () -> Content

    var body: some View {
        GeometryReader { geo in
            if geo.size.height > geo.size.width {
                VStack(spacing: spacing) { content() }
            } else {
                HStack(spacing: spacing) { content() }
            }
        }
    }
}
