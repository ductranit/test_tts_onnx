//
//  SpeechController.swift
//  Runner
//
//  Created by ductran on 14/07/2023.
//
import Foundation
import Flutter
import AVFoundation
import onnxruntime_objc

class SpeechController: NSObject, FlutterStreamHandler, FlutterPlugin {
    static let shared: SpeechController = SpeechController()
    var registrar: FlutterPluginRegistrar? = nil
    private var engine: AVAudioEngine?
    private var player: AVAudioPlayerNode?
    
    func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        return nil
    }
    
    func onCancel(withArguments arguments: Any?) -> FlutterError? {
        return nil
    }
    
    static func register(with registrar: FlutterPluginRegistrar) {
        let messenger = registrar.messenger()
        let channel = FlutterMethodChannel(
                 name: "tts",
                    binaryMessenger: messenger)


        let instance = SpeechController.shared
        instance.registrar = registrar
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    
    
    func generateAudio(lightspeechPath: String, mbmelganPath: String) throws{
        let ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        let lightspeechSession = try ORTSession(env: ortEnv, modelPath: lightspeechPath, sessionOptions: nil)
        let mbmelganSession = try ORTSession(env: ortEnv, modelPath: mbmelganPath, sessionOptions: nil)
        let lightspeech = LightSpeech()
        let mbmelgan = MBMelGAN()
        let lightspeechResults = lightspeech.infer(ortEnv: ortEnv, ortSession: lightspeechSession)
        let durations = lightspeechResults.durations[0]
        // infer melgan
            let mels = lightspeechResults.mels
            let mbmelganResults = mbmelgan.infer(ortEnv: ortEnv, ortSession: mbmelganSession, mels: mels)
            
            var durationString = ""
            for i in durations {
                durationString += "\(i) "
            }
        
        let audio = mbmelganResults.audio[0].flatMap { $0 }.map { Float($0) }
        let data = audio.withUnsafeBufferPointer { Data(buffer: $0) }
        playAudio(data: data)
//        let wrapper = TTSClassWrapper()
//        let result = wrapper.generateTTSAudio(withLightspeechPath: lightspeechPath, mbmelganPath: mbmelganPath)!
//        let floatArray = result.compactMap { $0 as? Float }
//        let data = floatArray.withUnsafeBufferPointer { Data(buffer: $0) }
//        let d = result.withUnsafeBufferPointer { Data(buffer: $0) }
//        playAudio(data: data)
    }
    
    func playAudio(data: Data) {
        let audioFormat = AVAudioFormat(standardFormatWithSampleRate: Double(44100), channels: 1)!
        guard let buffer = data.makePCMBuffer(format: audioFormat)  else{
            return
        }
        
        engine = AVAudioEngine()
        player = AVAudioPlayerNode()
        let mixer = engine!.mainMixerNode
        engine?.attach(player!)
        engine?.connect(player!, to: mixer, format: audioFormat)
        
        do {
            engine?.prepare()
            try engine?.start()
        } catch {
            print("Error info: \(error)")
        }
        
        player?.play()
        player?.scheduleBuffer(buffer) {
           print("complete")
        }
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
            case "run":
                print("")
            guard let lightspeech = Bundle.main.path(forResource: "lightspeech_quant", ofType: "onnx"), let mbmelgan = Bundle.main.path(forResource: "mbmelgan", ofType: "onnx") else {
               
                result(nil)
                return
            }
            
            try? generateAudio(lightspeechPath:lightspeech , mbmelganPath: mbmelgan)
            result(nil)
            default:
                result(FlutterMethodNotImplemented)
        }
    }
    
    public func detachFromEngineForRegistrar(registrar: FlutterPluginRegistrar) {
            print("SpeechController detachFromEngineForRegistrar")
            
        }
}

extension Data {
    init(buffer: AVAudioPCMBuffer, time: AVAudioTime) {
        let audioBuffer = buffer.audioBufferList.pointee.mBuffers
        self.init(bytes: audioBuffer.mData!, count: Int(audioBuffer.mDataByteSize))
    }

    func makePCMBuffer(format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let streamDesc = format.streamDescription.pointee
        let frameCapacity = UInt32(count) / streamDesc.mBytesPerFrame
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCapacity) else { return nil }

        buffer.frameLength = buffer.frameCapacity
        let audioBuffer = buffer.audioBufferList.pointee.mBuffers

        withUnsafeBytes { (bufferPointer) in
            guard let addr = bufferPointer.baseAddress else { return }
            audioBuffer.mData?.copyMemory(from: addr, byteCount: Int(audioBuffer.mDataByteSize))
        }

        return buffer
    }
}
