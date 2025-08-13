//
//  ChartView.swift
//  LinearAlgebraDemo
//
//  Created by Robert Ryan on 8/12/25.
//

import SwiftUI
import Charts

struct ChartView: View {
    let chartData: [DataPoint]

    var body: some View {
        VStack {
            Text("Inference Results")
                .font(.title)

            Chart {
                ForEach(chartData, id: \.self) { result in
                    BarMark(
                        x: .value("Name", result.name),
                        y: .value("Score", result.value)
                    )
                }
            }
            .chartYScale(domain: 0.0 ... 1.0)
            .chartYAxis {
                AxisMarks(values: Array(stride(from: 0.0, through: 1.0, by: 0.1))) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel {   // trailing alignment
                        if let doubleValue = value.as(Double.self) {
                            Text(doubleValue, format: .percent)
                                .frame(maxWidth: .infinity, alignment: .trailing)
                        }
                    }
                }
            }
        }
    }
}
