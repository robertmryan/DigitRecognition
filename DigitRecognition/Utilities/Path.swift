//
//  Path.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/22/25.
//

import SwiftUI

extension Collection where Element == CGPoint {
    func path() -> Path {
        if isEmpty { return Path() }

        var path = Path()
        var iterator = makeIterator()

        guard var priorPoint = iterator.next() else {
            return Path()
        }
        var count = 1
        path.move(to: priorPoint)
        while let point = iterator.next() {
            let mid = CGPoint(
                x: (point.x + priorPoint.x) / 2,
                y: (point.y + priorPoint.y) / 2
            )
            path.addQuadCurve(to: mid, control: priorPoint)
            priorPoint = point
            count += 1
        }

        if count > 1 {
            path.addLine(to: priorPoint)
        } else {
            path.addEllipse(in: CGRect(origin: priorPoint, size: .zero))
        }

        return path
    }
}
