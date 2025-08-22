//
//  IDXSequence.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/9/25.
//

import Foundation

/// An `AsyncSequence` used for iterating through a IDX file.

struct IDXSequence {
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
}

// MARK: - AsyncSequence

extension IDXSequence: AsyncSequence, AsyncIteratorProtocol {
    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(
            imagesBytes: imagesBytes,
            labelsBytes: labelsBytes,
            imagesHeader: imagesHeader,
            labelsHeader: labelsHeader
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

        private func readBytes(_ count: Int, from bytes: inout URL.AsyncBytes.AsyncIterator) async throws -> [UInt8]? {
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
}

// MARK: - Private implementation methods

private extension IDXSequence {
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

extension IDXSequence {
    enum IDXError: Error {
        case missingHeader
    }
}
