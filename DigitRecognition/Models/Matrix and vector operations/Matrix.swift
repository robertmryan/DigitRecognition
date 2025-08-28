//
//  Matrix.swift
//
//  Created by Robert Ryan on 8/12/25.
//

import Foundation
import Accelerate

/// Matrix
///
/// We are using Accelerate framework (notably cBLAS and vDSP) for vector/matrix calculations.
///
/// The backing storage here is obviously a simple array, so we’ll capture the number of rows
/// and columns so we ensure correct usage of this type with the appropriate preconditions.

struct Matrix<Element: Equatable>: ExpressibleByArrayLiteral {
    let rows: Int
    let cols: Int
    fileprivate(set) var buffer: Array<Element>

    init(_ elements: [[Element]]) {
        self.rows = elements.count
        self.cols = elements[0].count

        let count = rows * cols

        buffer = elements.flatMap { $0 }

        precondition(buffer.count == count, "All rows must have same number of columns")
    }

    init(elements: [Element], rows: Int, cols: Int) {
        precondition(rows * cols == elements.count)

        self.rows = rows
        self.cols = cols
        self.buffer = elements
    }

    init(repeating element: Element, rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        buffer = .init(repeating: element, count: rows * cols)
    }

    init(arrayLiteral elements: [Element]...) {
        rows = elements.count
        cols = elements[0].count

        buffer = elements.flatMap { $0 }

        precondition(buffer.count == count, "All rows must have same number of columns")
    }
}

// MARK: - Generic interfaces

extension Matrix {
    var count: Int { rows * cols }

    subscript(_ index: Int) -> Element {
        get { buffer[index] }
        set { buffer[index] = newValue }
    }

    subscript(_ row: Int, _ col: Int) -> Element {
        get { buffer[row * cols + col] }
        set { buffer[row * cols + col] = newValue }
    }

    @inlinable
    mutating func withUnsafeMutableBufferPointer<R, E>(_ body: (inout UnsafeMutableBufferPointer<Element>) throws(E) -> R) throws(E) -> R where E : Error {
        try buffer.withUnsafeMutableBufferPointer(body)
    }
}

// MARK: - Matrix<Float> Operators

extension Matrix where Element == Float {
    static func * (lhs: Matrix<Element>, rhs: Matrix<Element>) -> Matrix<Element> {
        precondition(lhs.cols == rhs.rows)

        var result = Matrix<Element>(repeating: 0, rows: lhs.rows, cols: rhs.cols)
        vDSP_mmul(
            lhs.buffer, 1,         // A is m × p
            rhs.buffer, 1,         // B is p × n
            &result.buffer, 1,     // C is m × n
            vDSP_Length(lhs.rows), // m
            vDSP_Length(rhs.cols), // n
            vDSP_Length(lhs.cols)  // p
        )
        return result
    }
}

// MARK: - Matrix<Double> operators

extension Matrix where Element == Double {
    static func * (lhs: Matrix<Element>, rhs: Matrix<Element>) -> Matrix<Element> {
        precondition(lhs.cols == rhs.rows)

        var result = Matrix<Element>(repeating: 0, rows: lhs.rows, cols: rhs.cols)
        vDSP_mmulD(
            lhs.buffer, 1,         // A is m × p
            rhs.buffer, 1,         // B is p × n
            &result.buffer, 1,     // C is m × n
            vDSP_Length(lhs.rows), // m
            vDSP_Length(rhs.cols), // n
            vDSP_Length(lhs.cols)  // p
        )
        return result
    }
}

// MARK: - Matrix<Float> implementations

extension Matrix where Element == Float {
    mutating func rowWiseUpdate(
        delta: Vector<Float>,
        prevAct: Vector<Float>,
        learningRate: Float
    ) {
        let alpha: Float = -learningRate
        // y := alpha * x * y^T + A
        cblas_sger(
            CblasRowMajor,
            rows,           // m
            cols,           // n
            alpha,
            delta.buffer,   // x
            1,              // incx
            prevAct.buffer, // y
            1,              // incy
            &buffer,
            cols            // A, lda = cols for row-major
        )
    }
}

// MARK: - Equatable

extension Matrix: Equatable where Element: Equatable {
    // Equatable conformance:
    static func == (lhs: Matrix<Element>, rhs: Matrix<Element>) -> Bool {
        guard lhs.rows == rhs.rows, lhs.cols == rhs.cols else { return false }
        return lhs.buffer.elementsEqual(rhs.buffer)
    }
}

// MARK: - CustomStringConvertible

extension Matrix: CustomStringConvertible {
    var description: String {
        var string = "Matrix<\(String(describing: Element.self))>([\n"
        for row in 0..<rows {
            string += "    ["
            for col in 0..<cols {
                let index = row * cols + col
                string += "\(buffer[index])"
                if col < cols - 1 {
                    string += ", "
                }
            }
            if row < rows - 1 {
                string += "],\n"
            } else {
                string += "]\n"
            }
        }
        string += "])"
        return string
    }
}

// MARK: - Vector<Float> implementations

// While this is a `Vector` method, it is updating the internal `buffer`
// of a matrix. Thus, that’s why it is here.

extension Vector where Element == Float {
    @inlinable
    func outerProduct(with b: Vector<Element>) -> Matrix<Element> {
        precondition(count == b.count)

        var result = Matrix<Element>(repeating: 0, rows: count, cols: b.count)
        vDSP_mmul(
            buffer, 1,                     // A is m × p (i.e., m × 1)
            b.buffer, 1,                   // B is p × n (i.e., 1 × n)
            &result.buffer, 1,             // C is m × n (i.e., m × n)
            vDSP_Length(count),            // m
            vDSP_Length(b.count),          // n
            vDSP_Length(1)                 // p
        )
        return result
    }
}

// MARK: - Vector<Double> implementations

// While this is a `Vector` method, it is updating the internal `buffer`
// of a matrix. Thus, that’s why it is here.

extension Vector where Element == Double {
    @inlinable
    func outerProduct(with b: Vector<Element>) -> Matrix<Element> {
        precondition(count == b.count)

        var result = Matrix<Element>(repeating: 0, rows: count, cols: b.count)
        vDSP_mmulD(
            buffer, 1,            // A is m × p (i.e., m × 1)
            b.buffer, 1,          // B is p × n (i.e., 1 × n)
            &result.buffer, 1,    // C is m × n (i.e., m × n)
            vDSP_Length(count),   // m
            vDSP_Length(b.count), // n
            vDSP_Length(1)        // p
        )
        return result
    }
}
