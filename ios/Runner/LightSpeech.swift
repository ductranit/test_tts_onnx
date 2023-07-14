//
//  LightSpeech.swift
//  Runner
//
//  Created by ductran on 14/07/2023.
//

import Foundation
import onnxruntime_objc

struct LightSpeechOutputs {
    let mels: [[[Float]]]
    let durations: [[Int32]]
}

class LightSpeech {
    
    func infer(ortEnv: ORTEnv, ortSession: ORTSession) -> LightSpeechOutputs {
        var result: LightSpeechOutputs
        
        // raw input vectors
        let inputIDs: [Int32] = [25, 29, 13, 40, 17, 51, 23, 29, 17, 12, 42, 16, 51, 14, 8, 51, 23, 3, 50, 4, 71, 68, 14, 29, 22, 50, 34, 29, 21, 25, 29, 4, 42, 21, 9, 29, 17, 17, 16, 51, 34, 33, 18, 17, 18, 47, 11, 33, 26, 8, 51, 13, 51, 14, 25, 29, 14, 39, 18, 72]
        let speakerIDs: [Int32] = [0]
        let speedRatios: [Float] = [1.0]
        let f0Ratios: [Float] = [1.0]
        let energyRatios: [Float] = [1.0]

        // this is the shape of the inputs, our equivalent to tf.expand_dims.
        let inputIDsShape: [NSNumber]  = [1, NSNumber(value: inputIDs.count)]
        let speakerIDsShape: [NSNumber] = [NSNumber(value: 1)]
        let speedRatiosShape: [NSNumber] = [NSNumber(value: 1)]
        let f0RatiosShape: [NSNumber] = [NSNumber(value: 1)]
        let energyRatiosShape: [NSNumber] = [NSNumber(value: 1)]

        let inputNames = ["input_ids", "speaker_ids", "speed_ratios", "f0_ratios", "energy_ratios"]

        // create input tensors from raw vectors
        let inputIDsTensor = try! ORTValue(tensorData: NSMutableData(bytes: inputIDs, length: inputIDs.count * MemoryLayout<Int32>.size), elementType: ORTTensorElementDataType.int32, shape: inputIDsShape)
        let speakerIDsTensor = try! ORTValue(tensorData: NSMutableData(bytes: speakerIDs, length: speakerIDs.count * MemoryLayout<Int32>.size), elementType: ORTTensorElementDataType.int32, shape: speakerIDsShape)
        let speedRatiosTensor = try! ORTValue(tensorData: NSMutableData(bytes: speedRatios, length: speedRatios.count * MemoryLayout<Float>.size), elementType: ORTTensorElementDataType.float, shape: speedRatiosShape)
        let f0RatiosTensor = try! ORTValue(tensorData: NSMutableData(bytes: f0Ratios, length: f0Ratios.count * MemoryLayout<Float>.size), elementType: ORTTensorElementDataType.float, shape: f0RatiosShape)
        let energyRatiosTensor = try! ORTValue(tensorData: NSMutableData(bytes: energyRatios, length: energyRatios.count * MemoryLayout<Float>.size), elementType: ORTTensorElementDataType.float, shape: energyRatiosShape)
        
        let inputTensors = [inputIDsTensor, speakerIDsTensor, speedRatiosTensor, f0RatiosTensor, energyRatiosTensor]

        // create input name -> input tensor map
        var inputMap: [String: ORTValue] = [:]
        for (index, name) in inputNames.enumerated() {
            inputMap[name] = inputTensors[index]
        }
        
        let output = try! ortSession.run(withInputs: inputMap, outputNames: ["Identity", "Identity_1", "Identity_2"], runOptions: nil)
        let mels = try! output["Identity"]!.tensorData()
        let durations = try! output["Identity_1"]!.tensorData()
        let durationShapeInfo = try! output["Identity_1"]?.tensorTypeAndShapeInfo()
        let melsShapeInfo = try! output["Identity"]?.tensorTypeAndShapeInfo()
        
        // Convert mels NSMutableData to [[[Float]]]
        let melsPointer = mels.bytes.assumingMemoryBound(to: Float.self)
        let melsDims = melsShapeInfo!.shape.map{ Int(truncating: $0) }

        var melsArray: [[[Float]]] = Array(repeating: Array(repeating: Array(repeating: 0.0, count: melsDims[2]), count: melsDims[1]), count: melsDims[0])

        for i in 0..<melsDims[0] {
            for j in 0..<melsDims[1] {
                for k in 0..<melsDims[2] {
                    melsArray[i][j][k] = melsPointer[i*melsDims[1]*melsDims[2] + j*melsDims[2] + k]
                }
            }
        }

        // Convert durations NSMutableData to [[Int32]]
        let durationsPointer = durations.bytes.assumingMemoryBound(to: Int32.self)
        let durationsDims = durationShapeInfo!.shape.map{ Int(truncating: $0) }

        var durationsArray: [[Int32]] = Array(repeating: Array(repeating: 0, count: durationsDims[1]), count: durationsDims[0])

        for i in 0..<durationsDims[0] {
            for j in 0..<durationsDims[1] {
                durationsArray[i][j] = durationsPointer[i*durationsDims[1] + j]
            }
        }
        
        // convert to array
        result = LightSpeechOutputs(mels: melsArray, durations: durationsArray)

        return result
    }
}
