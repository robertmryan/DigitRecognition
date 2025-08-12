//
//  LinearAlgebraTests.swift
//  LinearAlgebraTests
//
//  Created by Robert Ryan on 8/9/25.
//

import Testing
@testable import LinearAlgebraDemo

struct LinearAlgebraTestsFloat {
    typealias Scalar = Float

    @Test func matrixTimesVector() {
        let matrix = Matrix<Scalar>([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        let vector: Vector<Scalar> = Vector([1, 2])

        let result = LinearAlgebra.matrixTimesVector(matrix: matrix, vector: vector)
        let expectedResult = Vector<Scalar>([9, 12, 15])

        #expect(result == expectedResult)
    }

    @Test func vectorTimesMatrix() {
        let vector: Vector<Scalar> = Vector([1, 2, 3])
        let matrix = Matrix<Scalar>([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        let result = LinearAlgebra.vectorTimesMatrix(vector: vector, matrix: matrix)
        let expectedResult = Vector<Scalar>([14, 32])

        #expect(result == expectedResult)
    }

    @Test func matrixTimesMatrix() {
        let lhs: Matrix<Scalar> = Matrix([
            [1, 2, 3],
            [4, 5, 6],
        ])
        let rhs = Matrix<Scalar>([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        let result = LinearAlgebra.matrixTimesMatrix(lhs: lhs, rhs: rhs)
        let expectedResult = Matrix<Scalar>([
            [14, 32],
            [32, 77]
        ])

        #expect(result == expectedResult)
    }

    @Test func innerProduct() {
        let a: Vector<Scalar> = Vector(
            [1, 2, 3],
        )

        let b: Vector<Scalar> = Vector(
            [5, 6, 7],
        )

        let result = a.innerProduct(with: b)

        #expect(result == 5 + 12 + 21)
    }

    @Test func unitVector() {
        let vector: Vector<Scalar> = Vector(
            [1, 2, 3, 4, 5, 6],
        )
        let normalized = vector.unitVector()

        let result = normalized.innerProduct(with: normalized)
        let expectedResult: Scalar = 1

        #expect(abs(result - expectedResult) < 0.0001)
    }

    @Test
    func softmaxBasicProperties() {
        let logits = Vector<Scalar>([2.0, 1.0, 0.1])
        let probs = logits.softmax()

        // Sum should be approximately 1.0
        let sum = probs.sum()
        #expect(abs(sum - 1.0) < 1e-6, "Softmax probabilities must sum to 1, got \(sum)")

        // All probabilities > 0
        for p in probs {
            #expect(p > 0, "Softmax output must be positive, got \(p)")
        }
    }

    @Test
    func softmaxKnownOutput() {
        let logits = Vector<Scalar>([2.0, 1.0, 0.1])
        let expected = Vector<Scalar>([0.65900114, 0.24243297, 0.09856589])

        let output = logits.softmax()

        for (o, e) in zip(output, expected) {
            #expect(abs(o - e) < 1e-3, "Softmax output \(o) differs from expected \(e)")
        }
    }
}

struct LinearAlgebraTestsDouble {
    typealias Scalar = Double

    @Test func matrixTimesVector() {
        let matrix = Matrix<Scalar>([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        let vector: Vector<Scalar> = Vector([1, 2])

        let result = LinearAlgebra.matrixTimesVector(matrix: matrix, vector: vector)
        let expectedResult = Vector<Scalar>([9, 12, 15])

        #expect(result == expectedResult)
    }

    @Test func vectorTimesMatrix() {
        let vector: Vector<Scalar> = Vector([1, 2, 3])
        let matrix = Matrix<Scalar>([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        let result = LinearAlgebra.vectorTimesMatrix(vector: vector, matrix: matrix)
        let expectedResult = Vector<Scalar>([14, 32])

        #expect(result == expectedResult)
    }

    @Test func matrixTimesMatrix() {
        let lhs: Matrix<Scalar> = Matrix([
            [1, 2, 3],
            [4, 5, 6],
        ])
        let rhs = Matrix<Scalar>([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        let result = LinearAlgebra.matrixTimesMatrix(lhs: lhs, rhs: rhs)
        let expectedResult = Matrix<Scalar>([
            [14, 32],
            [32, 77]
        ])

        #expect(result == expectedResult)
    }

    @Test func innerProduct() {
        let a: Vector<Scalar> = Vector(
            [1, 2, 3],
        )

        let b: Vector<Scalar> = Vector(
            [5, 6, 7],
        )

        let result = a.innerProduct(with: b)

        #expect(result == 5 + 12 + 21)
    }

    @Test func unitVector() {
        let vector: Vector<Scalar> = Vector(
            [1, 2, 3, 4, 5, 6],
        )
        let normalized = vector.unitVector()

        let result = normalized.innerProduct(with: normalized)
        let expectedResult: Scalar = 1

        #expect(abs(result - expectedResult) < 0.0001)
    }

    @Test
    func softmaxBasicProperties() {
        let logits = Vector<Scalar>([2.0, 1.0, 0.1])
        let probs = logits.softmax()

        // Sum should be approximately 1.0
        let sum = probs.sum()
        #expect(abs(sum - 1.0) < 1e-6, "Softmax probabilities must sum to 1, got \(sum)")

        // All probabilities > 0
        for p in probs {
            #expect(p > 0, "Softmax output must be positive, got \(p)")
        }
    }

    @Test
    func softmaxKnownOutput() {
        let logits = Vector<Scalar>([2.0, 1.0, 0.1])
        let expected = Vector<Scalar>([0.65900114, 0.24243297, 0.09856589])

        let output = logits.softmax()

        for (o, e) in zip(output, expected) {
            #expect(abs(o - e) < 1e-3, "Softmax output \(o) differs from expected \(e)")
        }
    }
}
