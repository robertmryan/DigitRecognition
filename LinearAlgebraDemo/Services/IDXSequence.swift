//
//  IDXSequence.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/9/25.
//

import Foundation

class IDXSequence: AsyncSequence, AsyncIteratorProtocol {
    var imagesBytes: URL.AsyncBytes.AsyncIterator
    var labelsBytes: URL.AsyncBytes.AsyncIterator
    var imagesHeader: IDXHeader!
    var labelsHeader: IDXHeader!

    init(images: URL, labels: URL) async throws {
        imagesBytes = images.resourceBytes.makeAsyncIterator()
        labelsBytes = labels.resourceBytes.makeAsyncIterator()
        imagesHeader = try await readHeader(for: &imagesBytes)
        labelsHeader = try await readHeader(for: &labelsBytes)
    }

    func readHeader(for bytes: inout URL.AsyncBytes.AsyncIterator) async throws -> IDXHeader {
        _ = try await bytes.next()
        _ = try await bytes.next()

        guard
            let byte = try await bytes.next(),
            let type = IDXDataType(rawValue: byte)
        else {
            throw IDXError.missingHeader
        }

        guard let dimension = try await bytes.next() else {
            throw IDXError.missingHeader
        }

        var counts: [Int] = []
        var countPerItem = 1

        let count = Int(try await readUInt32(from: &bytes))

        for _ in 1..<dimension {
            let count = Int(try await readUInt32(from: &bytes))
            countPerItem *= count
            counts.append(count)
        }

        return IDXHeader(
            count: count,
            countPerItem: countPerItem,
            dimensions: dimension,
            type: type,
            counts: [Int(dimension)] + counts
        )
    }

    struct AsyncIterator: AsyncIteratorProtocol {
        var imagesBytes: URL.AsyncBytes.AsyncIterator
        var labelsBytes: URL.AsyncBytes.AsyncIterator
        var imagesHeader: IDXHeader
        var labelsHeader: IDXHeader
        var current = 0

        mutating func next() async throws -> IDXRecord? {
            guard
                let imageBytes = try await readBytes(imagesHeader.countPerItem, from: &imagesBytes),
                let labelBytes = try await readBytes(labelsHeader.countPerItem, from: &labelsBytes)
            else {
                return nil
            }

            return IDXRecord(imageBytes: imageBytes, labelBytes: labelBytes)
        }

        func readBytes(_ count: Int, from bytes: inout URL.AsyncBytes.AsyncIterator) async throws -> [UInt8]? {
            var result: [UInt8] = []
            for _ in 0..<count {
                guard let byte = try await bytes.next() else {
                    return nil
                }
                result.append(byte)
            }
            return result
        }
    }

    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(
            imagesBytes: imagesBytes,
            labelsBytes: labelsBytes,
            imagesHeader: imagesHeader,
            labelsHeader: labelsHeader
        )
    }

    func readUInt32(from bytes: inout URL.AsyncBytes.AsyncIterator) async throws -> UInt32 {
        guard
            let byte1 = try await bytes.next().flatMap({ UInt32($0) }),
            let byte2 = try await bytes.next().flatMap({ UInt32($0) }),
            let byte3 = try await bytes.next().flatMap({ UInt32($0) }),
            let byte4 = try await bytes.next().flatMap({ UInt32($0) })
        else {
            throw IDXError.missingHeader
        }

        return (byte1 << 24) | (byte2 << 16) | (byte3 << 8) | byte4
    }
}

struct IDXRecord {
    let imageBytes: [UInt8]
    let labelBytes: [UInt8]
}

struct IDXHeader {
    let count: Int
    let countPerItem: Int
    let dimensions: UInt8
    let type: IDXDataType
    let counts: [Int]
}

enum IDXDataType: UInt8 {
    case `unsigned` = 0x08 // unsigned byte
    case `signed` = 0x09   // signed byte
    case `short` = 0x0B    // short (2 bytes)
    case `int` = 0x0C      // int (4 bytes)
    case `float` = 0x0D    // float (4 bytes)
    case `double` = 0x0E   // double (8 bytes)
}

enum IDXError: Error {
    case missingHeader
}
