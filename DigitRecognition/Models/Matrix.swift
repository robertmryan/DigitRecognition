//
//  Matrix.swift
//
//  Created by Robert Ryan on 8/12/25.
//

import Foundation
import Accelerate

/// Matrix
///
/// We are using Accelerate framework (notably cBLAS and vDSP) for matrix calculations.
/// While there is a new vDSP interface (e.g., `vDsp.mmul` rather than the old `vDSP_mmul`,
/// to enjoy the cBLAS performance, we really need to use `UnsafeMutableBufferPointer`.
/// So, to simplify our call points, this will store the supplied array of values in a manually
/// allocated `UnsafeMutableBufferPointer` and clean it up in `deinit`.
///
/// The backing storage here is obviously a simple linear buffer, so we’ll capture the number
/// of rows and columns so we ensure correct usage of this type with the appropriate preconditions.

final class Matrix<Element: Equatable>: ExpressibleByArrayLiteral {
    let rows: Int
    let cols: Int
    let buffer: UnsafeMutableBufferPointer<Element>

    init(_ elements: [[Element]]) {
        self.rows = elements.count
        self.cols = elements[0].count

        let count = rows * cols

        // Flatten 2D array into 1D array
        let flatElements = elements.flatMap { $0 }

        precondition(flatElements.count == count, "All rows must have same number of columns")

        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: count)
        ptr.initialize(from: flatElements, count: count) // bulk initialize

        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    init(elements: [Element], rows: Int, cols: Int) {
        precondition(rows * cols == elements.count)

        self.rows = rows
        self.cols = cols

        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: elements.count)
        // Initialize elements from input array
        ptr.initialize(from: elements, count: elements.count)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: elements.count)
    }

    init(repeating: Element, rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        let count = rows * cols

        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: count)
        ptr.initialize(repeating: repeating, count: count)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    init(arrayLiteral elements: [Element]...) {
        self.rows = elements.count
        self.cols = elements[0].count

        let count = rows * cols

        // Flatten 2D array into 1D array
        let flatElements = elements.flatMap { $0 }

        precondition(flatElements.count == count, "All rows must have same number of columns")

        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: count)
        ptr.initialize(from: flatElements, count: count) // bulk initialize

        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    deinit {
        buffer.baseAddress?.deinitialize(count: buffer.count)
        buffer.baseAddress?.deallocate()
    }
}

extension Matrix {
    var count: Int { rows * cols }

    subscript(_ index: Int) -> Element {
        get { buffer[index] }
        set { buffer[index] = newValue }
    }
}

// MARK: - Matrix<Float> Operators

extension Matrix where Element == Float {
    static func * (lhs: Matrix<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.cols == rhs.count)

        let result = Vector<Element>(repeating: 0, count: lhs.rows)
        vDSP_mmul(
            lhs.buffer.baseAddress!, 1,    // A is m × p
            rhs.buffer.baseAddress!, 1,    // B is p × n
            result.buffer.baseAddress!, 1, // C is m × n
            vDSP_Length(lhs.rows),         // m
            1,                             // n
            vDSP_Length(lhs.cols)          // p
        )
        return result
    }

    static func * (lhs: Matrix<Element>, rhs: Matrix<Element>) -> Matrix<Element> {
        precondition(lhs.cols == rhs.rows)

        let result = Matrix<Element>(repeating: 0, rows: lhs.rows, cols: rhs.cols)
        vDSP_mmul(
            lhs.buffer.baseAddress!, 1,    // A is m × p
            rhs.buffer.baseAddress!, 1,    // B is p × n
            result.buffer.baseAddress!, 1, // C is m × n
            vDSP_Length(lhs.rows),         // m
            vDSP_Length(rhs.cols),         // n
            vDSP_Length(lhs.cols)          // p
        )
        return result
    }
}

// MARK: - Matrix<Double> operators

extension Matrix where Element == Double {
    static func * (lhs: Matrix<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.cols == rhs.count)

        let result = Vector<Element>(repeating: 0, count: lhs.rows)
        vDSP_mmulD(
            lhs.buffer.baseAddress!, 1,    // A is m × p
            rhs.buffer.baseAddress!, 1,    // B is p × n
            result.buffer.baseAddress!, 1, // C is m × n
            vDSP_Length(lhs.rows),         // m
            1,                             // n
            vDSP_Length(lhs.cols)          // p
        )
        return result
    }

    static func * (lhs: Matrix<Element>, rhs: Matrix<Element>) -> Matrix<Element> {
        precondition(lhs.cols == rhs.rows)

        let result = Matrix<Element>(repeating: 0, rows: lhs.rows, cols: rhs.cols)
        vDSP_mmulD(
            lhs.buffer.baseAddress!, 1,    // A is m × p
            rhs.buffer.baseAddress!, 1,    // B is p × n
            result.buffer.baseAddress!, 1, // C is m × n
            vDSP_Length(lhs.rows),         // m
            vDSP_Length(rhs.cols),         // n
            vDSP_Length(lhs.cols)          // p
        )
        return result
    }
}

extension Matrix where Element == Float {
    /// Multiply this matrix by x, add b, and scale by y.
    /// - Parameters:
    ///   - x: Vector to multiply with this matrix.
    ///   - b: Vector to add to that result.
    /// - Returns: The resulting `Vector`.

    @discardableResult
    func multiplied(
        by x: Vector<Element>,
        plus b: Vector<Element>,
    ) -> Vector<Element> {
        precondition(cols == x.count)
        precondition(rows == b.count)

        // 1. Compute logits z = W * x + b
        // z shape: (10,)
        let y = Vector(b)
        cblas_sgemv(
            CblasRowMajor,         // ORDER: Specifies row-major (C) or column-major (Fortran) data ordering.
            CblasNoTrans,          // TRANSA: Specifies whether to transpose matrix A.
            rows,                  // M: Number of rows in matrix A.
            cols,                  // N: Number of columns in matrix A.
            1,                     // ALPHA: Scaling factor for the product of matrix A and vector X.
            buffer.baseAddress!,   // A: Matrix A.
            cols,                  // LDA: The size of the first dimension of matrix A. For a matrix A[M][N] that uses column-major ordering, the value is the number of rows M. For a matrix that uses row-major ordering, the value is the number of columns N.
            x.buffer.baseAddress!, // X: Vector X.
            1,                     // INCX: Stride within X. For example, if incX is 7, every seventh element is used.
            1,                     // BETA: Scaling factor for vector Y.
            y.buffer.baseAddress,  // Y: Vector Y
            1                      // INCY: Stride within Y. For example, if incY is 7, every seventh element is used.
        )

        return y
    }
}

extension Matrix where Element == Double {
    /// Multiply this matrix by x, add b, and scale by y.
    /// - Parameters:
    ///   - x: Vector to multiply with this matrix.
    ///   - b: Vector to add to that result.
    /// - Returns: The resulting `Vector`.

    @discardableResult
    func multiplied(
        by x: Vector<Element>,
        plus b: Vector<Element>,
    ) -> Vector<Element> {
        precondition(cols == x.count)
        precondition(rows == b.count)

        // 1. Compute logits z = W * x + b
        // z shape: (10,)
        let y = Vector(b)
        cblas_dgemv(
            CblasRowMajor,         // ORDER: Specifies row-major (C) or column-major (Fortran) data ordering.
            CblasNoTrans,          // TRANSA: Specifies whether to transpose matrix A.
            rows,                  // M: Number of rows in matrix A.
            cols,                  // N: Number of columns in matrix A.
            1,                     // ALPHA: Scaling factor for the product of matrix A and vector X.
            buffer.baseAddress!,   // A: Matrix A.
            cols,                  // LDA: The size of the first dimension of matrix A. For a matrix A[M][N] that uses column-major ordering, the value is the number of rows M. For a matrix that uses row-major ordering, the value is the number of columns N.
            x.buffer.baseAddress!, // X: Vector X.
            1,                     // INCX: Stride within X. For example, if incX is 7, every seventh element is used.
            1,                     // BETA: Scaling factor for vector Y.
            y.buffer.baseAddress,  // Y: Vector Y
            1                      // INCY: Stride within Y. For example, if incY is 7, every seventh element is used.
        )

        return y
    }
}

extension Matrix: Equatable where Element: Equatable {
    // Equatable conformance:
    static func == (lhs: Matrix<Element>, rhs: Matrix<Element>) -> Bool {
        guard lhs.rows == rhs.rows, lhs.cols == rhs.cols else { return false }
        return lhs.buffer.elementsEqual(rhs.buffer)
    }
}

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
