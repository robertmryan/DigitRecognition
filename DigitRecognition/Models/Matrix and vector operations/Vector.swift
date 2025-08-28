//
//  Vector.swift
//
//  Created by Robert Ryan on 8/12/25.
//

import Foundation
import Accelerate

/// Vector
///
/// We are using Accelerate framework (notably cBLAS and vDSP) for vector/matrix calculations.

struct Vector<Element>: ExpressibleByArrayLiteral {
    var count: Int { buffer.count }

    fileprivate(set) var buffer: Array<Element>

    init(_ elements: [Element]) {
        buffer = elements
    }

    init(repeating element: Element, count: Int) {
        buffer = .init(repeating: element, count: count)
    }

    init(arrayLiteral elements: Element...) {
        buffer = elements
    }
}

// MARK: - Equatable conformance

extension Vector: Equatable where Element: Equatable {
    static func == (lhs: Vector<Element>, rhs: Vector<Element>) -> Bool {
        guard lhs.count == rhs.count else { return false }
        return lhs.buffer.elementsEqual(rhs.buffer)
    }
}

// MARK: - Generic interface

extension Vector {
    subscript(_ index: Int) -> Element {
        get { buffer[index] }
        set { buffer[index] = newValue }
    }

    @inlinable
    mutating func withUnsafeMutableBufferPointer<R, E>(_ body: (inout UnsafeMutableBufferPointer<Element>) throws(E) -> R) throws(E) -> R where E : Error {
        try buffer.withUnsafeMutableBufferPointer(body)
    }
}

// MARK: - Vector<Float> operators

extension Vector where Element == Float {
    @inlinable
    static func * (lhs: Vector, rhs: Matrix<Element>) -> Vector {
        precondition(lhs.count == rhs.rows)

        var result = Vector<Element>(repeating: 0, count: rhs.cols)
        vDSP_mmul(
            lhs.buffer, 1,         // A is m × p
            rhs.buffer, 1,         // B is p × n
            &result.buffer, 1,     // C is m × n
            1,                     // m
            vDSP_Length(rhs.cols), // n
            vDSP_Length(rhs.rows)  // p
        )
        return result
    }

    @inlinable
    static func * (lhs: Vector, rhs: Element) -> Vector {
        var result = Vector<Element>(repeating: 0, count: lhs.count)
        let rhs = Vector(arrayLiteral: rhs)

        vDSP_vsmul(
            lhs.buffer,
            1,
            rhs.buffer,
            &result.buffer,
            1,
            vDSP_Length(lhs.count)
        )

        return result
    }

    @inlinable
    static func + (lhs: Vector, rhs: Vector) -> Vector {
        precondition(lhs.count == rhs.count)

        var result = Vector<Element>(repeating: 0, count: rhs.count)
        vDSP_vadd(
            lhs.buffer,
            1,
            rhs.buffer,
            1,
            &result.buffer,
            1,
            vDSP_Length(lhs.count)
        )
        return result
    }

    @inlinable
    static func - (lhs: Vector, rhs: Vector) -> Vector {
        precondition(lhs.count == rhs.count)

        var result = Vector<Element>(repeating: 0, count: rhs.count)
        vDSP_vsub(
            rhs.buffer,
            1,
            lhs.buffer,
            1,
            &result.buffer,
            1,
            vDSP_Length(lhs.count)
        )
        return result
    }
}

// MARK: - Vector<Double> operators

extension Vector where Element == Double {
    @inlinable
    static func * (lhs: Vector<Element>, rhs: Matrix<Element>) -> Vector<Element> {
        precondition(lhs.count == rhs.rows)

        var result = Vector<Element>(repeating: 0, count: rhs.cols)
        vDSP_mmulD(
            lhs.buffer, 1,         // A is m × p
            rhs.buffer, 1,         // B is p × n
            &result.buffer, 1,     // C is m × n
            1,                     // m
            vDSP_Length(rhs.cols), // n
            vDSP_Length(rhs.rows)  // p
        )
        return result
    }

    @inlinable
    static func * (lhs: Vector, rhs: Element) -> Vector {
        var result = Vector<Element>(repeating: 0, count: lhs.count)
        let rhs = Vector(arrayLiteral: rhs)

        vDSP_vsmulD(
            lhs.buffer,
            1,
            rhs.buffer,
            &result.buffer,
            1,
            vDSP_Length(lhs.count)
        )

        return result
    }

    @inlinable
    static func + (lhs: Vector, rhs: Vector) -> Vector {
        precondition(lhs.count == rhs.count)

        var result = Vector<Element>(repeating: 0, count: rhs.count)
        vDSP_vaddD(
            lhs.buffer,
            1,
            rhs.buffer,
            1,
            &result.buffer,
            1,
            vDSP_Length(lhs.count)
        )
        return result
    }

    @inlinable
    static func - (lhs: Vector, rhs: Vector) -> Vector {
        precondition(lhs.count == rhs.count)

        var result = Vector<Element>(repeating: 0, count: rhs.count)
        vDSP_vsubD(
            rhs.buffer,
            1,
            lhs.buffer,
            1,
            &result.buffer,
            1,
            vDSP_Length(lhs.count)
        )
        return result
    }
}

// MARK: - Vector<Float> implementations

extension Vector where Element == Float {
    @inlinable
    func unitVector() -> Vector<Element> {
        var result = self
        // Step 1: Compute sum of squares
        var sumSquares: Float = 0
        vDSP_svesq(buffer, 1, &sumSquares, vDSP_Length(count))
        
        // Step 2: Euclidean norm = sqrt(sumSquares)
        var norm = sqrt(sumSquares)
        
        // Step 3: Normalize if norm > 0
        if norm > 0 {
            vDSP_vsdiv(buffer, 1, &norm, &result.buffer, 1, vDSP_Length(count))
        }
        return result
    }

    @inlinable
    func innerProduct(with b: Vector<Element>) -> Element {
        precondition(count == b.count)

        var result: Float = 0
        vDSP_mmul(
            buffer, 1,         // A is m × p (i.e., 1 × p)
            b.buffer, 1,       // B is p × n (i.e., p × 1)
            &result, 1,        // C is m × n (i.e., 1 × 1)
            vDSP_Length(1),    // m
            vDSP_Length(1),    // n
            vDSP_Length(count) // p
        )
        return result
    }

    @inlinable
    func max() -> Element {
        var result: Float = 0
        vDSP_maxv(buffer, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    func maxValueAndIndex() -> (value: Element, index: Int) {
        var value: Element = 0
        var index: Int = 0
        vDSP_maxvi(buffer, 1, &value, &index, vDSP_Length(count))
        return (value, index)
    }

    @inlinable
    func sum() -> Element {
        var result: Float = 0
        vDSP_sve(buffer, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    func softmax() -> Vector<Element> {
        var output = Vector<Element>(repeating: 0, count: count)

        // Step 1: Find max for numerical stability
        var negMaxVal = -max()

        // Step 2: Subtract maxVal from logits (in place)
        var shifted = Vector<Float>(repeating: 0, count: count)
        vDSP_vsadd(buffer, 1, &negMaxVal, &shifted.buffer, 1, vDSP_Length(count))

        // Step 3: Compute exponentials using vvexpf
        var countInt32 = Int32(count)
        vvexpf(&output.buffer, shifted.buffer, &countInt32)

        // Step 4: Sum exponentials
        var sumExp: Float = 0
        vDSP_sve(output.buffer, 1, &sumExp, vDSP_Length(count))

        // Step 5: Normalize by dividing by sumExp
        var invSumExp = 1 / sumExp
        vDSP_vsmul(output.buffer, 1, &invSumExp, &output.buffer, 1, vDSP_Length(count))

        return output
    }

    @inlinable
    func multiplied(by scalar: Element, plus vector: Vector<Element>) -> Vector<Element> {
        precondition(count == vector.count)

        var scalar = scalar

        var d = vector
        vDSP_vsma(
            buffer,            // A:  The input vector A in D = (A * B) + C.
            1,                 // IA: The distance between the elements in the input vector A.
            &scalar,           // B:  The input scalar value B in D = (A * B) + C.
            vector.buffer,     // C:  The input vector C in D = (A * B) + C.
            1,                 // IC: The distance between the elements in the input vector C.
            &d.buffer,         // D:  The output vector D in D = (A * B) + C.
            1,                 // ID: The distance between the elements in the output vector D.
            vDSP_Length(count) // N:  The number of elements that the function processes.
        )

        return d
    }

    @inlinable
    func multipliedInPlace(by scalar: Element, plus vector: inout Vector<Element>) {
        precondition(count == vector.count)

        var scalar = scalar

        vDSP_vsma(
            buffer,            // A:  The input vector A in D = (A * B) + C.
            1,                 // IA: The distance between the elements in the input vector A.
            &scalar,           // B:  The input scalar value B in D = (A * B) + C.
            vector.buffer,     // C:  The input vector C in D = (A * B) + C.
            1,                 // IC: The distance between the elements in the input vector C.
            &vector.buffer,    // D:  The output vector D in D = (A * B) + C.
            1,                 // ID: The distance between the elements in the output vector D.
            vDSP_Length(count) // N:  The number of elements that the function processes.
        )
    }

    @inlinable
    func multipliedInPlace(by scalar: Element, plus pointer: UnsafeMutablePointer<Element>) {
        var scalar = scalar

        vDSP_vsma(
            buffer,         // A:  The input vector A in D = (A * B) + C.
            1,                           // IA: The distance between the elements in the input vector A.
            &scalar,                     // B:  The input scalar value B in D = (A * B) + C.
            pointer,                     // C:  The input vector C in D = (A * B) + C.
            1,                           // IC: The distance between the elements in the input vector C.
            pointer,                     // D:  The output vector D in D = (A * B) + C. Overwrites C.
            1,                           // ID: The distance between the elements in the output vector D.
            vDSP_Length(count)           // N:  The number of elements that the function processes.
        )
    }

    /// In-place Hadamard product: self[i] *= other[i]
    mutating func formMultiplyInPlace(by other: Vector<Float>) {
        precondition(self.count == other.count)
        vDSP_vmul(
            buffer,
            1,
            other.buffer,
            1,
            &buffer,
            1,
            vDSP_Length(count)
        )
    }

    /// Rectified linear unit.
    ///
    /// Returns a NEW vector: out[i] = max(0, self[i])

    func relu() -> Vector<Float> {
        var result = Vector<Float>(repeating: 0, count: count)
        var zero: Float = 0
        vDSP_vthres(
            buffer,
            1,
            &zero,
            &result.buffer,
            1,
            vDSP_Length(count)
        )
        return result
    }

    /// ReLU prime
    ///
    /// First derivative of rectified linear unit function.
    ///
    /// Returns a NEW vector of ReLU'(z): 1 if z[i] > 0 else 0
    ///
    /// - Note: Most of these functions leverage Accelerate, but there isn’t a good stand-in for
    ///         this function, and the compiler will tend to do a pretty good job optimizing and
    ///         vectorizing this naive implementation.

    func reluPrime() -> Vector<Float> {
        var out = Vector<Float>(repeating: 0, count: count)
        for i in 0..<count { out[i] = self[i] > 0 ? 1 : 0 }
        return out
    }
}

// MARK: - Vector<Double> implementations

extension Vector where Element == Double {
    @inlinable
    func unitVector() -> Vector<Element> {
        var result = self
        // Step 1: Compute sum of squares
        var sumSquares: Double = 0
        vDSP_svesqD(buffer, 1, &sumSquares, vDSP_Length(count))

        // Step 2: Euclidean norm = sqrt(sumSquares)
        var norm = sqrt(sumSquares)

        // Step 3: Normalize if norm > 0
        if norm > 0 {
            vDSP_vsdivD(buffer, 1, &norm, &result.buffer, 1, vDSP_Length(count))
        }
        return result
    }

    @inlinable
    func innerProduct(with b: Vector<Element>) -> Element {
        precondition(count == b.count)

        var result: Double = 0
        vDSP_mmulD(
            buffer, 1,        // A is m × p (i.e., 1 × p)
            b.buffer, 1,      // B is p × n (i.e., p × 1)
            &result, 1,                    // C is m × n (i.e., 1 × 1)
            vDSP_Length(1),                // m
            vDSP_Length(1),                // n
            vDSP_Length(count)             // p
        )
        return result
    }

    @inlinable
    func max() -> Double {
        var result: Double = 0
        vDSP_maxvD(buffer, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    func sum() -> Double {
        var result: Double = 0
        vDSP_sveD(buffer, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    func softmax() -> Vector<Double> {
        var output = Vector<Double>(repeating: 0, count: count)

        // Step 1: Find max for numerical stability
        var negMaxVal = -max()

        // Step 2: Subtract maxVal from logits (in place)
        var shifted = Vector<Double>(repeating: 0, count: count)
        vDSP_vsaddD(buffer, 1, &negMaxVal, &shifted.buffer, 1, vDSP_Length(count))

        // Step 3: Compute exponentials using vvexpf
        var countInt32 = Int32(count)
        vvexp(&output.buffer, shifted.buffer, &countInt32)

        // Step 4: Sum exponentials
        var sumExp: Double = 0
        vDSP_sveD(output.buffer, 1, &sumExp, vDSP_Length(count))

        // Step 5: Normalize by dividing by sumExp
        var invSumExp = 1 / sumExp
        vDSP_vsmulD(output.buffer, 1, &invSumExp, &output.buffer, 1, vDSP_Length(count))

        return output
    }

    @inlinable
    func multiplied(by scalar: Element, plus vector: Vector<Element>) -> Vector<Element> {
        precondition(count == vector.count)

        var scalar = scalar

        var d = vector
        vDSP_vsmaD(
            buffer,            // A:  The input vector A in D = (A * B) + C.
            1,                 // IA: The distance between the elements in the input vector A.
            &scalar,           // B:  The input scalar value B in D = (A * B) + C.
            vector.buffer,     // C:  The input vector C in D = (A * B) + C.
            1,                 // IC: The distance between the elements in the input vector C.
            &d.buffer,         // D:  The output vector D in D = (A * B) + C.
            1,                 // ID: The distance between the elements in the output vector D.
            vDSP_Length(count) // N:  The number of elements that the function processes.
        )

        return d
    }

    @inlinable
    func multiplied(by scalar: Element, plus pointer: UnsafeMutablePointer<Element>) {
        var scalar = scalar

        vDSP_vsmaD(
            buffer,         // A:  The input vector A in D = (A * B) + C.
            1,                           // IA: The distance between the elements in the input vector A.
            &scalar,                     // B:  The input scalar value B in D = (A * B) + C.
            pointer,                     // C:  The input vector C in D = (A * B) + C.
            1,                           // IC: The distance between the elements in the input vector C.
            pointer,                     // D:  The output vector D in D = (A * B) + C. Overwrites C.
            1,                           // ID: The distance between the elements in the output vector D.
            vDSP_Length(count)           // N:  The number of elements that the function processes.
        )
    }
}

// MARK: - Sequence conformance

extension Vector: Sequence {
    func makeIterator() -> Iterator {
        return Iterator(vector: self)
    }

    struct Iterator: IteratorProtocol {
        let vector: Vector<Element>
        private var index: Int = 0

        init(vector: Vector<Element>) {
            self.vector = vector
        }

        mutating func next() -> Element? {
            if index >= vector.count { return nil }
            defer { index += 1 }
            return vector.buffer[index]
        }
    }
}

// MARK: - CustomStringConvertible

extension Vector: CustomStringConvertible {
    var description: String {
        var string = "Vector<\(String(describing: Element.self))>(["
        string += self.map { "\($0)" }.joined(separator: ", ")
        string += "])"
        return string
    }
}

// MARK: - Matrix<Float> implementations

// While these are `Matrix` methods, they are updating the internal `buffer`
// of a vector. Thus, that’s why it is here.

extension Matrix where Element == Float {
    static func * (lhs: Matrix<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.cols == rhs.count)

        var result = Vector<Element>(repeating: 0, count: lhs.rows)
        vDSP_mmul(
            lhs.buffer, 1,         // A is m × p
            rhs.buffer, 1,         // B is p × n
            &result.buffer, 1,     // C is m × n
            vDSP_Length(lhs.rows), // m
            1,                     // n
            vDSP_Length(lhs.cols)  // p
        )
        return result
    }

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
        var y = b
        cblas_sgemv(
            CblasRowMajor,         // ORDER: Specifies row-major (C) or column-major (Fortran) data ordering.
            CblasNoTrans,          // TRANSA: Specifies whether to transpose matrix A.
            rows,                  // M: Number of rows in matrix A.
            cols,                  // N: Number of columns in matrix A.
            1,                     // ALPHA: Scaling factor for the product of matrix A and vector X.
            buffer,                // A: Matrix A.
            cols,                  // LDA: The size of the first dimension of matrix A. For a matrix A[M][N] that uses column-major ordering, the value is the number of rows M. For a matrix that uses row-major ordering, the value is the number of columns N.
            x.buffer,              // X: Vector X.
            1,                     // INCX: Stride within X. For example, if incX is 7, every seventh element is used.
            1,                     // BETA: Scaling factor for vector Y.
            &y.buffer,             // Y: Vector Y
            1                      // INCY: Stride within Y. For example, if incY is 7, every seventh element is used.
        )

        return y
    }

    /// Compute v = W^T * u, where W is (out, in), u is (out), result is (in).
    func transposeMultiply(_ u: Vector<Float>) -> Vector<Float> {
        precondition(u.count == rows)

        let alpha: Float = 1
        let beta:  Float = 0

        var v = Vector<Float>(repeating: 0, count: cols)
        cblas_sgemv(
            CblasRowMajor, // matches your storage (row-major)
            CblasTrans,    // we want W^T * u
            rows,          // m = rows of A (W)
            cols,          // n = cols of A (W)
            alpha,
            buffer,        // A
            cols,          // lda = number of columns in row-major
            u.buffer,      // x
            1,             // incx
            beta,
            &v.buffer,     // y
            1              // incy
        )
        return v
    }
}

// MARK: - Matrix<Double> implementations

// While this is a `Matrix` method, it is updating the internal `buffer`
// of a vector. Thus, that’s why it is here.

extension Matrix where Element == Double {
    static func * (lhs: Matrix<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.cols == rhs.count)

        var result = Vector<Element>(repeating: 0, count: lhs.rows)
        vDSP_mmulD(
            lhs.buffer, 1,         // A is m × p
            rhs.buffer, 1,         // B is p × n
            &result.buffer, 1,     // C is m × n
            vDSP_Length(lhs.rows), // m
            1,                     // n
            vDSP_Length(lhs.cols)  // p
        )
        return result
    }

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
        var y = b
        cblas_dgemv(
            CblasRowMajor,         // ORDER: Specifies row-major (C) or column-major (Fortran) data ordering.
            CblasNoTrans,          // TRANSA: Specifies whether to transpose matrix A.
            rows,                  // M: Number of rows in matrix A.
            cols,                  // N: Number of columns in matrix A.
            1,                     // ALPHA: Scaling factor for the product of matrix A and vector X.
            buffer,                // A: Matrix A.
            cols,                  // LDA: The size of the first dimension of matrix A. For a matrix A[M][N] that uses column-major ordering, the value is the number of rows M. For a matrix that uses row-major ordering, the value is the number of columns N.
            x.buffer,              // X: Vector X.
            1,                     // INCX: Stride within X. For example, if incX is 7, every seventh element is used.
            1,                     // BETA: Scaling factor for vector Y.
            &y.buffer,             // Y: Vector Y
            1                      // INCY: Stride within Y. For example, if incY is 7, every seventh element is used.
        )

        return y
    }
}
