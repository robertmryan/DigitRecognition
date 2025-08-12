//
//  LinearAlgebraDemoApp.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/9/25.
//

import SwiftUI

@main
struct LinearAlgebraDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
            #if os(iPad)
                .statusBar(hidden: true)
            #endif
        }
    }
}
