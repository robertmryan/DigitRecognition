//
//  IDXHeader.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/22/25.
//

struct IDXHeader {
    let count: Int
    let countPerItem: Int
    let dimensions: UInt8
    let type: IDXDataType
    let counts: [Int]
}
