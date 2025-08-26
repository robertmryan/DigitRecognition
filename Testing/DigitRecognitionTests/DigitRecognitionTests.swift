//
//  DigitRecognitionTests.swift
//  DigitRecognitionTests
//
//  Created by Robert Ryan on 8/14/25.
//

import Testing
@testable import DigitRecognition

struct DigitRecognitionTestsFloat {
    typealias Scalar = Float

    @Test func matrixTimesVector() {
        let matrix: Matrix<Scalar> = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        let vector: Vector<Scalar> = [1, 2]

        let result = matrix * vector
        let expectedResult: Vector<Scalar> = [9, 12, 15]

        #expect(result == expectedResult)
    }

    @Test func vectorTimesMatrix() {
        let vector: Vector<Scalar> = [1, 2, 3]
        let matrix: Matrix<Scalar> = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        let result = vector * matrix
        let expectedResult: Vector<Scalar> = [14, 32]

        #expect(result == expectedResult)
    }

    @Test func matrixTimesMatrix() {
        let lhs: Matrix<Scalar> = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        let rhs: Matrix<Scalar> = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        let result = lhs * rhs
        let expectedResult: Matrix<Scalar> = [
            [14, 32],
            [32, 77]
        ]

        #expect(result == expectedResult)
    }

    @Test func vectorMax() {
        let vector: Vector<Scalar> = [1, 7, 3]
        let expectedResult: Scalar = 7
        #expect(vector.max() == expectedResult)
    }

    @Test func innerProduct() {
        let a: Vector<Scalar> = [1, 2, 3]
        let b: Vector<Scalar> = [5, 6, 7]

        let result = a.innerProduct(with: b)

        #expect(result == 5 + 12 + 21)
    }

    @Test func unitVector() {
        let vector: Vector<Scalar> = [1, 2, 3, 4, 5, 6]
        let normalized = vector.unitVector()

        let result = normalized.innerProduct(with: normalized)
        let expectedResult: Scalar = 1

        #expect(abs(result - expectedResult) < 0.0001)
    }

    @Test
    func softmaxBasicProperties() {
        let logits: Vector<Scalar> = [2.0, 1.0, 0.1]
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
        let logits: Vector<Scalar> = [2.0, 1.0, 0.1]
        let expected: Vector<Scalar> = [0.65900114, 0.24243297, 0.09856589]

        let output = logits.softmax()

        for (o, e) in zip(output, expected) {
            #expect(abs(o - e) < 1e-3, "Softmax output \(o) differs from expected \(e)")
        }
    }

    @Test
    func vectorAddition() {
        let vector1: Vector<Scalar> = [7, 23]
        let vector2: Vector<Scalar> = [1, 2]
        let expectedResult: Vector<Scalar> = [8, 25]
        let result = vector1 + vector2
        #expect(result == expectedResult)
    }

    @Test
    func vectorSubtraction() {
        let vector1: Vector<Scalar> = [7, 23]
        let vector2: Vector<Scalar> = [1, 2]
        let expectedResult: Vector<Scalar> = [6, 21]
        let result = vector1 - vector2
        #expect(result == expectedResult)
    }

    @Test
    func vectorScaled() {
        let vector: Vector<Scalar> = [7, 23]
        let b: Scalar = 2
        let expectedResult: Vector<Scalar> = [14, 46]
        let result = vector * b
        #expect(result == expectedResult)
    }

    @Test
    func matrixMultipliedByVectorAndVectorAdded() {
        let matrix: Matrix<Scalar> = [
            [1, 2],
            [3, 4]
        ]
        let vector1: Vector<Scalar> = [2, 7]
        let vector2: Vector<Scalar> = [4, 2]
        let expectedResult = matrix * vector1 + vector2
        let result = matrix.multiplied(by: vector1, plus: vector2)
        #expect(result == expectedResult)
    }

    @Test
    func vectorScaledPlusVector() {
        let a: Vector<Scalar> = [1, 2]
        let b: Scalar = 2
        let c: Vector<Scalar> = [3, 4]

        let expectedResult: Vector<Scalar> = [5, 8]
        let result = a.multiplied(by: b, plus: c)
        #expect(result == expectedResult)
    }

    @Test
    func counting() {
        let vector: Vector<Scalar> = [1, 2]
        let matrix: Matrix<Scalar> = [
            [1, 2],
            [3, 4]
        ]

        #expect(vector.count == 2)
        #expect(matrix.count == 4)
    }

    @Test
    func subscripting() {
        let vector: Vector<Scalar> = [1, 2]
        let matrix: Matrix<Scalar> = [
            [1, 2],
            [3, 4]
        ]

        #expect(vector[0] == 1)
        #expect(vector[1] == 2)
        #expect(matrix[0] == 1)
        #expect(matrix[1] == 2)
        #expect(matrix[2] == 3)
        #expect(matrix[3] == 4)
    }

    @Test
    func multiplyScalarByPlusVector() {
        let vector1: Vector<Scalar> = [1, 2]
        let scalar: Scalar = 3
        let vector2: Vector<Scalar> = [4, 5]

        vector1.multiplied(by: scalar, plus: vector2.buffer.baseAddress!)
        let expectedResults: Vector<Scalar> = [
            (1 * 3) + 4,
            (2 * 3) + 5,
        ]

        #expect(vector2 == expectedResults)
    }

    @Test
    func multiplyVectorByPlusVector() {
        let matrix: Matrix<Scalar> = [
            [1, 2],
            [3, 4]
        ]
        let vector1: Vector<Scalar> = [5, 6]
        let vector2: Vector<Scalar> = [7, 8]

        let results = matrix.multiplied(by: vector1, plus: vector2)
        let expectedResults: Vector<Scalar> = [
            (1 * 5) + (2 * 6) + 7,
            (3 * 5) + (4 * 6) + 8,
        ]

        #expect(results == expectedResults)
    }

    @Test
    func unequalVectorsOfDifferentSizes() {
        let vector1: Vector<Scalar> = [1, 2]
        let vector2: Vector<Scalar> = [1, 2, 3]

        #expect(vector1 != vector2)
    }

    @Test
    func subscriptSetter() {
        let vector1: Vector<Scalar> = [1, 2]
        let vector2: Vector<Scalar> = [1, 3]
        vector1[1] = 3

        #expect(vector1 == vector2)
    }

    @Test
    func outerProduct() {
        let vector1: Vector<Scalar> = [1, 2]
        let vector2: Vector<Scalar> = [3, 4]
        let results = vector1.outerProduct(with: vector2)
        let expectedResults: Matrix<Scalar> = [
            [1 * 3, 1 * 4],
            [2 * 3, 2 * 4]
        ]

        #expect(results == expectedResults)
    }

    @Test
    func vectorDescription() async throws {
        let vector: Vector<Scalar> = [1, 2]
        #expect("\(vector)" == "Vector<Float>([1.0, 2.0])")
    }

    @Test
    func matrixDescription() async throws {
        let matrix: Matrix<Scalar> = [
            [1, 2],
            [3, 4]
        ]
        let expectedResults = """
            Matrix<Float>([
                [1.0, 2.0],
                [3.0, 4.0]
            ])
            """
        #expect("\(matrix)" == expectedResults)
    }

    @Test
    func matrixInitializerArrayOfArrays() async throws {
        let matrix1: Matrix<Scalar> = [
            [1, 2],
            [3, 4]
        ]
        let matrix2 = Matrix<Scalar>([
            [1, 2],
            [3, 4]
        ])
        #expect(matrix1 == matrix2)
    }

    @Test
    func matrixInitializerArray() async throws {
        let matrix1: Matrix<Scalar> = [
            [1, 2],
            [3, 4]
        ]
        let matrix2 = Matrix<Scalar>(
            elements: [
                1, 2,
                3, 4
            ],
            rows: 2,
            cols: 2
        )
        #expect(matrix1 == matrix2)
    }

    @Test
    func matrixSubscriptSetter() async throws {
        let matrix1: Matrix<Scalar> = [
            [1, 2],
            [3, 42]
        ]
        let matrix2 = Matrix<Scalar>([
            [1, 2],
            [3, 4]
        ])
        matrix1[3] = 4
        #expect(matrix1 == matrix2)
    }

    @Test
    func matrixInequality() async throws {
        let matrix1: Matrix<Scalar> = [
            [1, 2],
            [4, 5]
        ]
        let matrix2: Matrix<Scalar> = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        #expect(matrix1 != matrix2)
    }
}

struct DigitRecognitionTestsDouble {
    typealias Scalar = Double

    @Test func matrixTimesVector() {
        let matrix: Matrix<Scalar> = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        let vector: Vector<Scalar> = [1, 2]

        let result = matrix * vector
        let expectedResult: Vector<Scalar> = [9, 12, 15]

        #expect(result == expectedResult)
    }

    @Test func vectorTimesMatrix() {
        let vector: Vector<Scalar> = [1, 2, 3]
        let matrix: Matrix<Scalar> = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        let result = vector * matrix
        let expectedResult: Vector<Scalar> = [14, 32]

        #expect(result == expectedResult)
    }

    @Test func matrixTimesMatrix() {
        let lhs: Matrix<Scalar> = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        let rhs: Matrix<Scalar> = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        let result = lhs * rhs
        let expectedResult: Matrix<Scalar> = [
            [14, 32],
            [32, 77]
        ]

        #expect(result == expectedResult)
    }

    @Test func vectorMax() {
        let vector: Vector<Scalar> = [1, 7, 3]
        let expectedResult: Scalar = 7
        #expect(vector.max() == expectedResult)
    }

    @Test func innerProduct() {
        let a: Vector<Scalar> = [1, 2, 3]
        let b: Vector<Scalar> = [5, 6, 7]

        let result = a.innerProduct(with: b)

        #expect(result == 5 + 12 + 21)
    }

    @Test func unitVector() {
        let vector: Vector<Scalar> = [1, 2, 3, 4, 5, 6]
        let normalized = vector.unitVector()

        let result = normalized.innerProduct(with: normalized)
        let expectedResult: Scalar = 1

        #expect(abs(result - expectedResult) < 0.0001)
    }

    @Test
    func softmaxBasicProperties() {
        let logits:  Vector<Scalar> = [2.0, 1.0, 0.1]
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
        let logits: Vector<Scalar> = [2.0, 1.0, 0.1]
        let expected: Vector<Scalar> = [0.65900114, 0.24243297, 0.09856589]

        let output = logits.softmax()

        for (o, e) in zip(output, expected) {
            #expect(abs(o - e) < 1e-3, "Softmax output \(o) differs from expected \(e)")
        }
    }

    @Test
    func vectorAddition() {
        let vector1: Vector<Scalar> = [7, 23]
        let vector2: Vector<Scalar> = [1, 2]
        let expectedResult: Vector<Scalar> = [8, 25]
        let result = vector1 + vector2
        #expect(result == expectedResult)
    }

    @Test
    func vectorSubtraction() {
        let vector1: Vector<Scalar> = [7, 23]
        let vector2: Vector<Scalar> = [1, 2]
        let expectedResult: Vector<Scalar> = [6, 21]
        let result = vector1 - vector2
        #expect(result == expectedResult)
    }

    @Test
    func vectorScaled() {
        let vector: Vector<Scalar> = [7, 23]
        let b: Scalar = 2
        let expectedResult: Vector<Scalar> = [14, 46]
        let result = vector * b
        #expect(result == expectedResult)
    }

    @Test
    func matrixMultipliedByVectorAndVectorAdded() {
        let matrix: Matrix<Scalar> = [
            [1, 2],
            [3, 4]
        ]
        let vector1: Vector<Scalar> = [2, 7]
        let vector2: Vector<Scalar> = [4, 2]
        let expectedResult = matrix * vector1 + vector2
        let result = matrix.multiplied(by: vector1, plus: vector2)
        #expect(result == expectedResult)
    }

    @Test
    func vectorScaledPlusVector() {
        let a: Vector<Scalar> = [1, 2]
        let b: Scalar = 2
        let c: Vector<Scalar> = [3, 4]

        let expectedResult: Vector<Scalar> = [5, 8]
        let result = a.multiplied(by: b, plus: c)
        #expect(result == expectedResult)
    }

    @Test
    func multiplyScalarByPlusVector() {
        let vector1: Vector<Scalar> = [1, 2]
        let scalar: Scalar = 3
        let vector2: Vector<Scalar> = [4, 5]

        vector1.multiplied(by: scalar, plus: vector2.buffer.baseAddress!)
        let expectedResults: Vector<Scalar> = [
            (1 * 3) + 4,
            (2 * 3) + 5,
        ]

        #expect(vector2 == expectedResults)
    }

    @Test
    func multiplyVectorByPlusVector() {
        let matrix: Matrix<Scalar> = [
            [1, 2],
            [3, 4]
        ]
        let vector1: Vector<Scalar> = [5, 6]
        let vector2: Vector<Scalar> = [7, 8]

        let results = matrix.multiplied(by: vector1, plus: vector2)
        let expectedResults: Vector<Scalar> = [
            (1 * 5) + (2 * 6) + 7,
            (3 * 5) + (4 * 6) + 8,
        ]

        #expect(results == expectedResults)
    }

    @Test
    func outerProduct() {
        let vector1: Vector<Scalar> = [1, 2]
        let vector2: Vector<Scalar> = [3, 4]
        let results = vector1.outerProduct(with: vector2)
        let expectedResults: Matrix<Scalar> = [
            [1 * 3, 1 * 4],
            [2 * 3, 2 * 4]
        ]

        #expect(results == expectedResults)
    }
}
