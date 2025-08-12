//
//  LinearAlgebra.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/9/25.
//

import Foundation
import Accelerate

final class Matrix<T: Equatable> {
    let rows: Int
    let cols: Int
    let buffer: UnsafeMutableBufferPointer<T>

    init(_ elements: [[T]]) {
        self.rows = elements.count
        self.cols = elements[0].count

        let count = rows * cols

        // Flatten 2D array into 1D array
        let flatElements = elements.flatMap { $0 }

        precondition(flatElements.count == count, "All rows must have same number of columns")

        let ptr = UnsafeMutablePointer<T>.allocate(capacity: count)
        ptr.initialize(from: flatElements, count: count) // bulk initialize

        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    init(elements: [T], rows: Int, cols: Int) {
        precondition(rows * cols == elements.count)

        self.rows = rows
        self.cols = cols

        let ptr = UnsafeMutablePointer<T>.allocate(capacity: elements.count)
        // Initialize elements from input array
        ptr.initialize(from: elements, count: elements.count)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: elements.count)
    }

    init(repeating: T, rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        let count = rows * cols

        let ptr = UnsafeMutablePointer<T>.allocate(capacity: count)
        ptr.initialize(repeating: repeating, count: count)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    deinit {
        buffer.baseAddress?.deinitialize(count: buffer.count)
        buffer.baseAddress?.deallocate()
    }
}

extension Matrix {
    func withUnsafeBufferPointer(_ body: (UnsafeBufferPointer<T>) throws -> Void) rethrows {
        try body(UnsafeBufferPointer(buffer))
    }

    func withUnsafeMutableBufferPointer(_ body: (UnsafeMutableBufferPointer<T>) throws -> Void) rethrows {
        try body(buffer)
    }

    var count: Int { rows * cols }

    subscript(_ index: Int) -> T {
        get { buffer[index] }
        set { buffer[index] = newValue }
    }
}

extension Matrix: Equatable where T: Equatable {
    // Equatable conformance:
    static func == (lhs: Matrix<T>, rhs: Matrix<T>) -> Bool {
        guard lhs.rows == rhs.rows, lhs.cols == rhs.cols else { return false }
        return lhs.buffer.elementsEqual(rhs.buffer)
    }
}

extension Matrix: CustomStringConvertible {
    var description: String {
        var string = "Matrix<\(String(describing: T.self))>([\n"
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
                string += "], \n"
            } else {
                string += "]\n"
            }
        }
        string += "])"
        return string
    }
}

final class Vector<T> {
    let count: Int
    let buffer: UnsafeMutableBufferPointer<T>

    init(_ elements: [T]) {
        count = elements.count

        let ptr = UnsafeMutablePointer<T>.allocate(capacity: count)
        ptr.initialize(from: elements, count: count) // bulk initialize

        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    init(_ other: Vector<T>) {
        self.count = other.count
        let ptr = UnsafeMutablePointer<T>.allocate(capacity: count)
        ptr.initialize(from: other.buffer.baseAddress!, count: count)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    init(repeating: T, count: Int) {
        self.count = count
        let ptr = UnsafeMutablePointer<T>.allocate(capacity: count)
        ptr.initialize(repeating: repeating, count: count)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    deinit {
        buffer.baseAddress?.deinitialize(count: buffer.count)
        buffer.baseAddress?.deallocate()
    }
}

extension Vector: Equatable where T: Equatable {
    static func == (lhs: Vector<T>, rhs: Vector<T>) -> Bool {
        guard lhs.count == rhs.count else { return false }
        return lhs.buffer.elementsEqual(rhs.buffer)
    }
}

extension Vector {
    func withUnsafeBufferPointer(_ body: (UnsafeBufferPointer<T>) throws -> Void) rethrows {
        try body(UnsafeBufferPointer(buffer))
    }

    func withUnsafeMutableBufferPointer(_ body: (UnsafeMutableBufferPointer<T>) throws -> Void) rethrows {
        try body(buffer)
    }

    subscript(_ index: Int) -> T {
        get { buffer[index] }
        set { buffer[index] = newValue }
    }
}

extension Vector where T == Float {
    func unitVector() -> Vector<T> {
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

    func innerProduct(with b: Vector<Float>) -> Float {
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

    func max() -> Float {
        var result: Float = 0
        vDSP_maxv(buffer.baseAddress!, 1, &result, vDSP_Length(count))
        return result
    }

    func sum() -> Float {
        var result: Float = 0
        vDSP_sve(buffer.baseAddress!, 1, &result, vDSP_Length(count))
        return result
    }

    func softmax() -> Vector<Float> {
        let output = Vector<Float>(repeating: 0, count: count)

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
}

extension Vector where T == Double {
    func unitVector() -> Vector<T> {
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

    func innerProduct(with b: Vector<Double>) -> Double {
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

    func max() -> Double {
        var result: Double = 0
        vDSP_maxvD(buffer.baseAddress!, 1, &result, vDSP_Length(count))
        return result
    }

    func sum() -> Double {
        var result: Double = 0
        vDSP_sveD(buffer.baseAddress!, 1, &result, vDSP_Length(count))
        return result
    }

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
}

extension Vector: Sequence {
    func makeIterator() -> Iterator {
        return Iterator(vector: self)
    }

    struct Iterator: IteratorProtocol {
        let vector: Vector<T>
        private var index: Int = 0

        init(vector: Vector<T>) {
            self.vector = vector
        }

        mutating func next() -> T? {
            if index >= vector.count { return nil }
            defer { index += 1 }
            return vector.buffer[index]
        }
    }
}

extension Vector: CustomStringConvertible {
    var description: String {
        var string = "Vector<\(String(describing: T.self))>(["
        string += self.map { "\($0)" }.joined(separator: ", ")
        string += "])"
        return string
    }
}

struct LinearAlgebra { }

// MARK: - Float implementations

extension LinearAlgebra {
    static func vectorTimesMatrix(vector: Vector<Float>, matrix: Matrix<Float>) -> Vector<Float> {
        precondition(vector.count == matrix.rows)

        let result = Vector<Float>(repeating: 0, count: matrix.cols)
        vDSP_mmul(
            vector.buffer.baseAddress!, 1, // A is m × p
            matrix.buffer.baseAddress!, 1, // B is p × n
            result.buffer.baseAddress!, 1, // C is m × n
            1,                             // m
            vDSP_Length(matrix.cols),      // n
            vDSP_Length(matrix.rows)       // p
        )
        return result
    }

    static func matrixTimesVector(matrix: Matrix<Float>, vector: Vector<Float>) -> Vector<Float> {
        precondition(matrix.cols == vector.count)

        let result = Vector<Float>(repeating: 0, count: matrix.rows)
        vDSP_mmul(
            matrix.buffer.baseAddress!, 1, // A is m × p
            vector.buffer.baseAddress!, 1, // B is p × n
            result.buffer.baseAddress!, 1, // C is m × n
            vDSP_Length(matrix.rows),      // m
            1,                             // n
            vDSP_Length(matrix.cols)       // p
        )
        return result
    }

    static func matrixTimesMatrix(lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
        precondition(lhs.cols == rhs.rows)

        let result = Matrix<Float>(repeating: 0, rows: lhs.rows, cols: rhs.cols)
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

    static func test(
        x: Vector<Float>,       // Input vector (784,)
        w: Matrix<Float>,       // Weight matrix (10*784), row-major
        b: Vector<Float>,       // Bias vector (10,)
    ) -> Vector<Float> {
        let numClasses = w.rows
        let inputSize = x.count

        // 1. Compute logits z = W * x + b
        // z shape: (10,)
        let y = Vector<Float>(repeating: 0, count: numClasses)
        cblas_sgemv(
            CblasRowMajor,         // ORDER: Specifies row-major (C) or column-major (Fortran) data ordering.
            CblasNoTrans,          // TRANSA: Specifies whether to transpose matrix A.
            Int32(numClasses),     // M: Number of rows in matrix A.
            Int32(inputSize),      // N: Number of columns in matrix A.
            1,                     // ALPHA: Scaling factor for the product of matrix A and vector X.
            w.buffer.baseAddress!, // A: Matrix A.
            Int32(inputSize),      // LDA: The size of the first dimension of matrix A. For a matrix A[M][N] that uses column-major ordering, the value is the number of rows M. For a matrix that uses row-major ordering, the value is the number of columns N.
            x.buffer.baseAddress!, // X: Vector X.
            1,                     // INCX: Stride within X. For example, if incX is 7, every seventh element is used.
            0,                     // BETA: Scaling factor for vector Y.
            y.buffer.baseAddress,  // Y: Vector Y
            1                      // INCY: Stride within Y. For example, if incY is 7, every seventh element is used.
        )
        // Add bias
        vDSP_vadd(b.buffer.baseAddress!, 1, y.buffer.baseAddress!, 1, y.buffer.baseAddress!, 1, vDSP_Length(numClasses))

        return y.softmax()
    }

    static func trainStep(
        x: Vector<Float>,       // Input vector (784,)
        t: Vector<Float>,       // Target one-hot vector (10,)
        w: Matrix<Float>,       // Weight matrix (10*784), row-major
        b: Vector<Float>,       // Bias vector (10,)
        learningRate: Float
    ) {
        let numClasses = t.count
        let inputSize = x.count

        // 1. Compute logits z = W * x + b
        // z shape: (10,)
        let z = Vector<Float>(repeating: 0, count: numClasses)
        cblas_sgemv(
            CblasRowMajor,         // ORDER: Specifies row-major (C) or column-major (Fortran) data ordering.
            CblasNoTrans,          // TRANSA: Specifies whether to transpose matrix A.
            Int32(numClasses),     // M: Number of rows in matrix A.
            Int32(inputSize),      // N: Number of columns in matrix A.
            1,                     // ALPHA: Scaling factor for the product of matrix A and vector X.
            w.buffer.baseAddress!, // A: Matrix A.
            Int32(inputSize),      // LDA: The size of the first dimension of matrix A. For a matrix A[M][N] that uses column-major ordering, the value is the number of rows M. For a matrix that uses row-major ordering, the value is the number of columns N.
            x.buffer.baseAddress!, // X: Vector X.
            1,                     // INCX: Stride within X. For example, if incX is 7, every seventh element is used.
            0,                     // BETA: Scaling factor for vector Y.
            z.buffer.baseAddress,  // Y: Vector Y
            1                      // INCY: Stride within Y. For example, if incY is 7, every seventh element is used.
        )
        // Add bias
        vDSP_vadd(b.buffer.baseAddress!, 1, z.buffer.baseAddress!, 1, z.buffer.baseAddress!, 1, vDSP_Length(numClasses))

        // 2. Compute softmax probabilities y
        let y = z.softmax()

        // 3. Compute error (y - t)
        let error = Vector<Float>(repeating: 0, count: numClasses)
        vDSP_vsub(t.buffer.baseAddress!, 1, y.buffer.baseAddress!, 1, error.buffer.baseAddress!, 1, vDSP_Length(numClasses)) // error = y - t

        // 4. Compute gradient for W: grad_W = error * x^T
        // grad_W shape: (10, 784)
        // We'll do outer product of error (10,) and x (784,)
        // Update weights: W = W - learningRate * grad_W
        for i in 0..<numClasses {
            let gradRowStart = i * inputSize
            let gradRowStartAddress = w.buffer.baseAddress!.advanced(by: gradRowStart)
            // grad_W[i, :] = error[i] * x[:]
            let scaledX = Vector<Float>(repeating: 0, count: inputSize)
            var scale = error[i]
            vDSP_vsmul(x.buffer.baseAddress!, 1, &scale, scaledX.buffer.baseAddress!, 1, vDSP_Length(inputSize))

            // W[i, :] -= learningRate * grad_W[i, :]
            var negLR = -learningRate
            vDSP_vsma(
                scaledX.buffer.baseAddress!, // A:  The input vector A in D = (A * B) + C.
                1,                           // IA: The distance between the elements in the input vector A.
                &negLR,                      // B:  The input scalar value B in D = (A * B) + C.
                gradRowStartAddress,         // C:  The input vector C in D = (A * B) + C.
                1,                           // IC: The distance between the elements in the input vector C.
                gradRowStartAddress,         // D:  The output vector D in D = (A * B) + C.
                1,                           // ID: The distance between the elements in the output vector D.
                vDSP_Length(inputSize)       // N:  The number of elements that the function processes.
            )
        }

        // 5. Update bias: b = b - learningRate * error
        var negLR = -learningRate
        vDSP_vsma(error.buffer.baseAddress!, 1, &negLR, b.buffer.baseAddress!, 1, b.buffer.baseAddress!, 1, vDSP_Length(numClasses))
    }
}

// MARK: - Double implementations

extension LinearAlgebra {
    static func vectorTimesMatrix(vector: Vector<Double>, matrix: Matrix<Double>) -> Vector<Double> {
        precondition(vector.count == matrix.rows)

        let result = Vector<Double>(repeating: 0, count: matrix.cols)
        vDSP_mmulD(
            vector.buffer.baseAddress!, 1, // A is m × p
            matrix.buffer.baseAddress!, 1, // B is p × n
            result.buffer.baseAddress!, 1, // C is m × n
            1,                             // m
            vDSP_Length(matrix.cols),      // n
            vDSP_Length(matrix.rows)       // p
        )
        return result
    }

    static func matrixTimesVector(matrix: Matrix<Double>, vector: Vector<Double>) -> Vector<Double> {
        precondition(matrix.cols == vector.count)

        let result = Vector<Double>(repeating: 0, count: matrix.rows)
        vDSP_mmulD(
            matrix.buffer.baseAddress!, 1, // A is m × p
            vector.buffer.baseAddress!, 1, // B is p × n
            result.buffer.baseAddress!, 1, // C is m × n
            vDSP_Length(matrix.rows),      // m
            1,                             // n
            vDSP_Length(matrix.cols)       // p
        )
        return result
    }

    static func matrixTimesMatrix(lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
        precondition(lhs.cols == rhs.rows)

        let result = Matrix<Double>(repeating: 0, rows: lhs.rows, cols: rhs.cols)
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
