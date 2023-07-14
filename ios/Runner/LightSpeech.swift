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
    let durations: [[Int]]
}

class LightSpeech {
    
    func infer(ortEnv: ORTEnv, ortSession: ORTSession) -> LightSpeechOutputs {
        var result: LightSpeechOutputs
        
        // raw input vectors
        let inputIDs = [25, 29, 13, 40, 17, 51, 23, 29, 17, 12, 42, 16, 51, 14, 8, 51, 23, 3, 50, 4, 71, 68, 14, 29, 22, 50, 34, 29, 21, 25, 29, 4, 42, 21, 9, 29, 17, 17, 16, 51, 34, 33, 18, 17, 18, 47, 11, 33, 26, 8, 51, 13, 51, 14, 25, 29, 14, 39, 18, 72]
        let speakerIDs = [0]
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
        let inputIDsTensor = try! ORTValue(tensorData: NSMutableData(bytes: inputIDs, length: inputIDs.count * MemoryLayout<Int>.size), elementType: ORTTensorElementDataType.int32, shape: inputIDsShape)
        let speakerIDsTensor = try! ORTValue(tensorData: NSMutableData(bytes: speakerIDs, length: speakerIDs.count * MemoryLayout<Int>.size), elementType: ORTTensorElementDataType.int32, shape: speakerIDsShape)
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
        
        let melsArr: [[[Float]]] = Array(_immutableCocoaArray: mels)
        let durationsArr: [[Int]] = Array(_immutableCocoaArray: durations)
        
        // convert to array
        result = LightSpeechOutputs(mels: melsArr, durations: durationsArr)

        return result
    }
}
