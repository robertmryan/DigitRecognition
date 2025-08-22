//
//  IDXDataType.swift
//  DigitRecognition
//
//  Created by Robert Ryan on 8/22/25.
//


enum IDXDataType: UInt8 {
    case `unsigned` = 0x08 // unsigned byte
    case `signed` = 0x09   // signed byte
    case `short` = 0x0B    // short (2 bytes)
    case `int` = 0x0C      // int (4 bytes)
    case `float` = 0x0D    // float (4 bytes)
    case `double` = 0x0E   // double (8 bytes)
}
