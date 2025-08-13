//
//  ViewModel+SwiftUI.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/12/25.
//

import SwiftUI

enum ViewModelError: LocalizedError {
    case cannotCreateImage

    var errorDescription: String? {
        switch self {
            case .cannotCreateImage: "Cannot create image"
        }
    }
}

extension ViewModel {
    nonisolated func cgImage(for imageData: [Float]) throws -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard
            let context = CGContext(
                data: nil,
                width: 28,
                height: 28,
                bitsPerComponent: 8,
                bytesPerRow: 28 * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue
            ),
            let buffer = context.data
        else {
            throw ViewModelError.cannotCreateImage
        }

        let pixelBuffer = buffer.bindMemory(to: UInt8.self, capacity: 28 * 4 * 28 * 4)
        for (index, pixel) in imageData.enumerated() {
            print(UInt8(255 * (1 - pixel)))
            let offset = index * 4
            pixelBuffer[offset] = UInt8(255 * (1 - pixel))
            pixelBuffer[offset + 1] = UInt8(255 * (1 - pixel))
            pixelBuffer[offset + 2] = UInt8(255 * (1 - pixel))
            pixelBuffer[offset + 3] = 255
        }

        guard let cgImage = context.makeImage() else {
            throw ViewModelError.cannotCreateImage
        }

        return cgImage
    }

    nonisolated func createImage(index: Int, from imageData: [Float]) throws {
#if os(iOS)
        let cgImage = try cgImage(for: imageData)
        let image = UIImage(cgImage: cgImage)
        guard let data = image.pngData() else {
            throw ViewModelError.cannotCreateImage
        }

        let folder = URL.temporaryDirectory.appending(path: "MNIST images")
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        print(folder)
        let fileUrl = folder.appendingPathComponent("\(index).png")
        try? FileManager.default.removeItem(at: fileUrl)
        try? data.write(to: fileUrl)

        //        let mutableData = CFDataCreateMutable(nil, 0)!
        //        guard let destination = CGImageDestinationCreateWithData(mutableData, UTType.jpeg.identifier as CFString, 1, nil) else {
        //            return
        //        }
        //
        //        let options = [
        //            kCGImageDestinationLossyCompressionQuality: 0.9
        //        ] as CFDictionary
        //
        //        CGImageDestinationAddImage(destination, cgImage, options)
        //        guard CGImageDestinationFinalize(destination) else {
        //            return
        //        }
        //
        //        let data = mutableData as Data
        //
        //        let folder = URL.temporaryDirectory.appending(path: "MNIST images")
        //        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        //        print(folder)
        //        let fileUrl = folder.appendingPathComponent("\(index).jpg")
        //        try? FileManager.default.removeItem(at: fileUrl)
        //        try? data.write(to: fileUrl)

        //        let nsImage = NSImage(cgImage: cgImage, size: CGSize(width: 28, height: 28))
        //        if let data = nsImage.tiffRepresentation {
        //            let folder = URL.temporaryDirectory.appending(path: "MNIST images")
        //            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        //            print(folder)
        //            let fileUrl = folder.appendingPathComponent("\(index).tiff")
        //            try? FileManager.default.removeItem(at: fileUrl)
        //            try? data.write(to: fileUrl)
        //        }
#endif
    }
}
