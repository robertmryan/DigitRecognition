//
//  GrayPixelBufferPool.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/25/25.
//

import CoreVideo

final class GrayPixelBufferPool {
    private var pool: CVPixelBufferPool!
    private let width: Int
    private let height: Int
    private let count: Int

    init(width: Int = 28, height: Int = 28) {
        self.width = width
        self.height = height
        self.count = width * height

        let attributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent8,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            // These two help ensure CoreVideo creates buffers we can use everywhere
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            // This key is often required when creating pixel buffers (even empty)
            kCVPixelBufferIOSurfacePropertiesKey: [:],
        ]
        let status = CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
        precondition(status == kCVReturnSuccess, "Could not create CVPixelBufferPool")
    }

    /// Copies bytes row-by-row respecting destination stride.
    /// `bufferPointer` must be `width * height` grayscale bytes.
    func make(_ bufferPointer: UnsafeRawBufferPointer) -> CVPixelBuffer {
        precondition(bufferPointer.count == count)
        var pixelBuffer: CVPixelBuffer!
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer)

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        let destBase = CVPixelBufferGetBaseAddress(pixelBuffer)!.assumingMemoryBound(to: UInt8.self)
        let destBPR = CVPixelBufferGetBytesPerRow(pixelBuffer)       // >= width
        let srcBase = bufferPointer.bindMemory(to: UInt8.self).baseAddress!

        if destBPR == width {
            // Fast path: tightly packed; one memcpy is fine
            memcpy(destBase, srcBase, count)
        } else {
            // General path: copy each row
            for y in 0..<height {
                let srcRow = srcBase.advanced(by: y * width)
                let dstRow = destBase.advanced(by: y * destBPR)
                memcpy(dstRow, srcRow, width)
            }
        }
        return pixelBuffer
    }
}
