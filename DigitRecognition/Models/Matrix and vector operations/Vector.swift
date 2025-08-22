//
//  Vector.swift
//
//  Created by Robert Ryan on 8/12/25.
//

import Foundation
import Accelerate

/// Vector
///
/// We are using Accelerate framework (notably cBLAS and vDSP) for matrix calculations.
/// While there is a new vDSP interface (e.g., `vDsp.mmul` rather than the old `vDSP_mmul`,
/// to enjoy the cBLAS performance, we really need to use `UnsafeMutableBufferPointer`.
/// So, to simplify our call points, this will store the supplied array of values in a manually
/// allocated `UnsafeMutableBufferPointer` and clean it up in `deinit`.

final class Vector<Element>: ExpressibleByArrayLiteral {
    let count: Int
    let buffer: UnsafeMutableBufferPointer<Element>

    init(_ elements: [Element]) {
        count = elements.count

        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: count)
        ptr.initialize(from: elements, count: count) // bulk initialize

        buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    init(_ other: Vector<Element>) {
        self.count = other.count
        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: count)
        ptr.initialize(from: other.buffer.baseAddress!, count: count)
        buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    init(repeating: Element, count: Int) {
        self.count = count
        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: count)
        ptr.initialize(repeating: repeating, count: count)
        buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    deinit {
        buffer.baseAddress?.deinitialize(count: buffer.count)
        buffer.baseAddress?.deallocate()
    }

    init(arrayLiteral elements: Element...) {
        count = elements.count

        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: count)
        ptr.initialize(from: elements, count: count) // bulk initialize

        buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }
}

// MARK: - Equatable conformance

extension Vector: Equatable where Element: Equatable {
    static func == (lhs: Vector<Element>, rhs: Vector<Element>) -> Bool {
        guard lhs.count == rhs.count else { return false }
        return lhs.buffer.elementsEqual(rhs.buffer)
    }
}

// MARK: - Subscript access

extension Vector {
    subscript(_ index: Int) -> Element {
        get { buffer[index] }
        set { buffer[index] = newValue }
    }
}

// MARK: - Vector<Float> operators

extension Vector where Element == Float {
    @inlinable
    static func * (lhs: Vector, rhs: Matrix<Element>) -> Vector {
        precondition(lhs.count == rhs.rows)

        let result = Vector<Element>(repeating: 0, count: rhs.cols)
        vDSP_mmul(
            lhs.buffer.baseAddress!, 1,    // A is m × p
            rhs.buffer.baseAddress!, 1,    // B is p × n
            result.buffer.baseAddress!, 1, // C is m × n
            1,                             // m
            vDSP_Length(rhs.cols),         // n
            vDSP_Length(rhs.rows)          // p
        )
        return result
    }

    @inlinable
    static func * (lhs: Vector, rhs: Element) -> Vector {
        let result = Vector<Element>(repeating: 0, count: lhs.count)
        let rhs = Vector(arrayLiteral: rhs)

        vDSP_vsmul(
            lhs.buffer.baseAddress!,
            1,
            rhs.buffer.baseAddress!,
            result.buffer.baseAddress!,
            1,
            vDSP_Length(lhs.count)
        )

        return result
    }

    @inlinable
    static func + (lhs: Vector, rhs: Vector) -> Vector {
        precondition(lhs.count == rhs.count)

        let result = Vector<Element>(repeating: 0, count: rhs.count)
        vDSP_vadd(
            lhs.buffer.baseAddress!,
            1,
            rhs.buffer.baseAddress!,
            1,
            result.buffer.baseAddress!,
            1,
            vDSP_Length(lhs.count)
        )
        return result
    }

    @inlinable
    static func - (lhs: Vector, rhs: Vector) -> Vector {
        precondition(lhs.count == rhs.count)

        let result = Vector<Element>(repeating: 0, count: rhs.count)
        vDSP_vsub(
            rhs.buffer.baseAddress!,
            1,
            lhs.buffer.baseAddress!,
            1,
            result.buffer.baseAddress!,
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

        let result = Vector<Element>(repeating: 0, count: rhs.cols)
        vDSP_mmulD(
            lhs.buffer.baseAddress!, 1, // A is m × p
            rhs.buffer.baseAddress!, 1, // B is p × n
            result.buffer.baseAddress!, 1, // C is m × n
            1,                             // m
            vDSP_Length(rhs.cols),      // n
            vDSP_Length(rhs.rows)       // p
        )
        return result
    }

    @inlinable
    static func * (lhs: Vector, rhs: Element) -> Vector {
        let result = Vector<Element>(repeating: 0, count: lhs.count)
        let rhs = Vector(arrayLiteral: rhs)

        vDSP_vsmulD(
            lhs.buffer.baseAddress!,
            1,
            rhs.buffer.baseAddress!,
            result.buffer.baseAddress!,
            1,
            vDSP_Length(lhs.count)
        )

        return result
    }

    @inlinable
    static func + (lhs: Vector, rhs: Vector) -> Vector {
        precondition(lhs.count == rhs.count)

        let result = Vector<Element>(repeating: 0, count: rhs.count)
        vDSP_vaddD(
            lhs.buffer.baseAddress!,
            1,
            rhs.buffer.baseAddress!,
            1,
            result.buffer.baseAddress!,
            1,
            vDSP_Length(lhs.count)
        )
        return result
    }

    @inlinable
    static func - (lhs: Vector, rhs: Vector) -> Vector {
        precondition(lhs.count == rhs.count)

        let result = Vector<Element>(repeating: 0, count: rhs.count)
        vDSP_vsubD(
            rhs.buffer.baseAddress!,
            1,
            lhs.buffer.baseAddress!,
            1,
            result.buffer.baseAddress!,
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
        let result = Vector(self)
        // Step 1: Compute sum of squares
        var sumSquares: Float = 0
        vDSP_svesq(buffer.baseAddress!, 1, &sumSquares, vDSP_Length(count))
        
        // Step 2: Euclidean norm = sqrt(sumSquares)
        var norm = sqrt(sumSquares)
        
        // Step 3: Normalize if norm > 0
        if norm > 0 {
            vDSP_vsdiv(buffer.baseAddress!, 1, &norm, result.buffer.baseAddress!, 1, vDSP_Length(count))
        }
        return result
    }

    @inlinable
    func innerProduct(with b: Vector<Element>) -> Element {
        precondition(count == b.count)

        var result: Float = 0
        vDSP_mmul(
            buffer.baseAddress!, 1,        // A is m × p (i.e., 1 × p)
            b.buffer.baseAddress!, 1,      // B is p × n (i.e., p × 1)
            &result, 1,                    // C is m × n (i.e., 1 × 1)
            vDSP_Length(1),                // m
            vDSP_Length(1),                // n
            vDSP_Length(count)             // p
        )
        return result
    }

    @inlinable
    func outerProduct(with b: Vector<Element>) -> Matrix<Element> {
        precondition(count == b.count)

        let result = Matrix<Element>(repeating: 0, rows: count, cols: b.count)
        vDSP_mmul(
            buffer.baseAddress!, 1,        // A is m × p (i.e., m × 1)
            b.buffer.baseAddress!, 1,      // B is p × n (i.e., 1 × n)
            result.buffer.baseAddress!, 1, // C is m × n (i.e., m × n)
            vDSP_Length(count),            // m
            vDSP_Length(b.count),          // n
            vDSP_Length(1)                 // p
        )
        return result
    }

    @inlinable
    func max() -> Element {
        var result: Float = 0
        vDSP_maxv(buffer.baseAddress!, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    func maxValueAndIndex() -> (value: Element, index: Int) {
        var value: Element = 0
        var index: Int = 0
        vDSP_maxvi(buffer.baseAddress!, 1, &value, &index, vDSP_Length(count))
        return (value, index)
    }

    @inlinable
    func sum() -> Element {
        var result: Float = 0
        vDSP_sve(buffer.baseAddress!, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    func softmax() -> Vector<Element> {
        let output = Vector<Element>(repeating: 0, count: count)

        // Step 1: Find max for numerical stability
        var negMaxVal = -max()

        // Step 2: Subtract maxVal from logits (in place)
        let shifted = Vector<Float>(repeating: 0, count: count)
        vDSP_vsadd(buffer.baseAddress!, 1, &negMaxVal, shifted.buffer.baseAddress!, 1, vDSP_Length(count))

            // Step 3: Compute exponentials using vvexpf
            var countInt32 = Int32(count)
        vvexpf(output.buffer.baseAddress!, shifted.buffer.baseAddress!, &countInt32)

            // Step 4: Sum exponentials
            var sumExp: Float = 0
        vDSP_sve(output.buffer.baseAddress!, 1, &sumExp, vDSP_Length(count))

            // Step 5: Normalize by dividing by sumExp
            var invSumExp = 1 / sumExp
        vDSP_vsmul(output.buffer.baseAddress!, 1, &invSumExp, output.buffer.baseAddress!, 1, vDSP_Length(count))

        return output
    }

    @inlinable
    func multiplied(by scalar: Element, plus vector: Vector<Element>) -> Vector<Element> {
        precondition(count == vector.count)

        var scalar = scalar

        let d = Vector(vector)
        vDSP_vsma(
            buffer.baseAddress!,         // A:  The input vector A in D = (A * B) + C.
            1,                           // IA: The distance between the elements in the input vector A.
            &scalar,                     // B:  The input scalar value B in D = (A * B) + C.
            vector.buffer.baseAddress!,  // C:  The input vector C in D = (A * B) + C.
            1,                           // IC: The distance between the elements in the input vector C.
            d.buffer.baseAddress!,       // D:  The output vector D in D = (A * B) + C.
            1,                           // ID: The distance between the elements in the output vector D.
            vDSP_Length(count)           // N:  The number of elements that the function processes.
        )

        return d
    }

    @inlinable
    func multiplied(by scalar: Element, plus pointer: UnsafeMutablePointer<Element>) {
        var scalar = scalar

        vDSP_vsma(
            buffer.baseAddress!,         // A:  The input vector A in D = (A * B) + C.
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
    func formMultiplyInPlace(by other: Vector<Float>) {
        precondition(self.count == other.count)
        vDSP_vmul(
            buffer.baseAddress!,
            1,
            other.buffer.baseAddress!,
            1,
            buffer.baseAddress!,
            1,
            vDSP_Length(count)
        )
    }

    /// Rectified linear unit.
    ///
    /// Returns a NEW vector: out[i] = max(0, self[i])

    func relu() -> Vector<Float> {
        let result = Vector<Float>(repeating: 0, count: count)
        var zero: Float = 0
        vDSP_vthres(
            buffer.baseAddress!,
            1,
            &zero,
            result.buffer.baseAddress!,
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
        let out = Vector<Float>(repeating: 0, count: count)
        for i in 0..<count { out[i] = self[i] > 0 ? 1 : 0 }
        return out
    }
}

// MARK: - Vector<Double> implementations

extension Vector where Element == Double {
    @inlinable
    func unitVector() -> Vector<Element> {
        let result = Vector(self)
        // Step 1: Compute sum of squares
        var sumSquares: Double = 0
        vDSP_svesqD(buffer.baseAddress!, 1, &sumSquares, vDSP_Length(count))

        // Step 2: Euclidean norm = sqrt(sumSquares)
        var norm = sqrt(sumSquares)

        // Step 3: Normalize if norm > 0
        if norm > 0 {
            vDSP_vsdivD(buffer.baseAddress!, 1, &norm, result.buffer.baseAddress!, 1, vDSP_Length(count))
        }
        return result
    }

    @inlinable
    func innerProduct(with b: Vector<Element>) -> Element {
        precondition(count == b.count)

        var result: Double = 0
        vDSP_mmulD(
            buffer.baseAddress!, 1,        // A is m × p (i.e., 1 × p)
            b.buffer.baseAddress!, 1,      // B is p × n (i.e., p × 1)
            &result, 1,                    // C is m × n (i.e., 1 × 1)
            vDSP_Length(1),                // m
            vDSP_Length(1),                // n
            vDSP_Length(count)             // p
        )
        return result
    }

    @inlinable
    func outerProduct(with b: Vector<Element>) -> Matrix<Element> {
        precondition(count == b.count)

        let result = Matrix<Element>(repeating: 0, rows: count, cols: b.count)
        vDSP_mmulD(
            buffer.baseAddress!, 1,        // A is m × p (i.e., m × 1)
            b.buffer.baseAddress!, 1,      // B is p × n (i.e., 1 × n)
            result.buffer.baseAddress!, 1, // C is m × n (i.e., m × n)
            vDSP_Length(count),            // m
            vDSP_Length(b.count),          // n
            vDSP_Length(1)                 // p
        )
        return result
    }

    @inlinable
    func max() -> Double {
        var result: Double = 0
        vDSP_maxvD(buffer.baseAddress!, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    func sum() -> Double {
        var result: Double = 0
        vDSP_sveD(buffer.baseAddress!, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    func softmax() -> Vector<Double> {
        let output = Vector<Double>(repeating: 0, count: count)

        // Step 1: Find max for numerical stability
        var negMaxVal = -max()

        // Step 2: Subtract maxVal from logits (in place)
        let shifted = Vector<Double>(repeating: 0, count: count)
        vDSP_vsaddD(buffer.baseAddress!, 1, &negMaxVal, shifted.buffer.baseAddress!, 1, vDSP_Length(count))

        // Step 3: Compute exponentials using vvexpf
        var countInt32 = Int32(count)
        vvexp(output.buffer.baseAddress!, shifted.buffer.baseAddress!, &countInt32)

        // Step 4: Sum exponentials
        var sumExp: Double = 0
        vDSP_sveD(output.buffer.baseAddress!, 1, &sumExp, vDSP_Length(count))

        // Step 5: Normalize by dividing by sumExp
        var invSumExp = 1 / sumExp
        vDSP_vsmulD(output.buffer.baseAddress!, 1, &invSumExp, output.buffer.baseAddress!, 1, vDSP_Length(count))

        return output
    }

    @inlinable
    func multiplied(by scalar: Element, plus vector: Vector<Element>) -> Vector<Element> {
        precondition(count == vector.count)

        var scalar = scalar

        let d = Vector(vector)
        vDSP_vsmaD(
            buffer.baseAddress!,         // A:  The input vector A in D = (A * B) + C.
            1,                           // IA: The distance between the elements in the input vector A.
            &scalar,                     // B:  The input scalar value B in D = (A * B) + C.
            vector.buffer.baseAddress!,  // C:  The input vector C in D = (A * B) + C.
            1,                           // IC: The distance between the elements in the input vector C.
            d.buffer.baseAddress!,       // D:  The output vector D in D = (A * B) + C.
            1,                           // ID: The distance between the elements in the output vector D.
            vDSP_Length(count)           // N:  The number of elements that the function processes.
        )

        return d
    }

    @inlinable
    func multiplied(by scalar: Element, plus pointer: UnsafeMutablePointer<Element>) {
        var scalar = scalar

        vDSP_vsmaD(
            buffer.baseAddress!,         // A:  The input vector A in D = (A * B) + C.
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
