//
//  MBMelGAN.swift
//  Runner
//
//  Created by ductran on 14/07/2023.
//

import Foundation
import onnxruntime_objc

struct MBMelGANOutputs {
    let audio: [[[Float]]]
}

class MBMelGAN {
    
    func infer(ortEnv: ORTEnv, ortSession: ORTSession, mels: [[[Float]]]) -> MBMelGANOutputs {
        var result: MBMelGANOutputs
        
        // unpack 3d FloatArray and get size along each dimension := (1, L, 80)
        let melsShape: [NSNumber] = [NSNumber(value: mels.count), NSNumber(value: mels[0].count), NSNumber(value: mels[0][0].count)]
        
        let totalElements = mels.count * mels[0].count * mels[0][0].count
        var flattenedMels = [Float](repeating: 0, count: totalElements)
        for i in 0..<mels.count {
            for j in 0..<mels[0].count {
                for k in 0..<mels[0][0].count {
                    let index = i * mels[0].count * mels[0][0].count + j * mels[0][0].count + k
                    flattenedMels[index] = mels[i][j][k]
                }
            }
        }
        
        // create input tensors from raw vectors
        let melTensor = try! ORTValue(tensorData: NSMutableData(bytes: flattenedMels, length: flattenedMels.count * MemoryLayout<Float>.size), elementType: ORTTensorElementDataType.float, shape: melsShape)
        
        // create input name -> input tensor map
        let inputTensors: [String: ORTValue] = ["mels": melTensor]

        let output = try! ortSession.run(withInputs: inputTensors, outputNames: [], runOptions: nil)
        let audio = output["0"]!.tensorData as! [[[Float]]]
        result = MBMelGANOutputs(audio: audio)

        return result
    }
}
