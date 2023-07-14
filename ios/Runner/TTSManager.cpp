//
//  TTSManager.cpp
//  Runner
//
//  Created by ductran on 14/07/2023.
//

#include "TTSManager.hpp"
#include <onnxruntime_cxx_api.h>

std::vector<float> generateTTSAudio(const char* lightspeech_path, const char* mbmelgan_path) {
        Ort::Env env;
        Ort::SessionOptions session_options;
        Ort::RunOptions run_options;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        Ort::Session lightspeech = Ort::Session(env, lightspeech_path, session_options);
        Ort::Session mbmelgan = Ort::Session(env, mbmelgan_path, session_options);

        // raw input vectors
        std::vector<int32_t> input_ids = {25, 29, 13, 40, 17, 51, 23, 29, 17, 12, 42, 16, 51, 14, 8, 51, 23, 3, 50, 4, 71, 68, 14, 29, 22, 50, 34, 29, 21, 25, 29, 4, 42, 21, 9, 29, 17, 17, 16, 51, 34, 33, 18, 17, 18, 47, 11, 33, 26, 8, 51, 13, 51, 14, 25, 29, 14, 39, 18, 72};
        std::vector<float> energy_ratios = {1.f};
        std::vector<float> f0_ratios = {1.f};
        // TODO: change speaker index here
        std::vector<int32_t> speaker_ids = {0};
        std::vector<float> speed_ratios = {1.f};

        // This is the shape of the inputs, our equivalent to tf.expand_dims.
        std::vector<int64_t> input_ids_shape = {1, (int64_t)input_ids.size()};
        std::vector<int64_t> energy_ratios_shape = {1};
        std::vector<int64_t> f0_ratios_shape = {1};
        std::vector<int64_t> speed_ratios_shape = {1};
        std::vector<int64_t> speaker_ids_shape = {1};

        const char* input_names[] = {"input_ids", "speaker_ids", "speed_ratios", "f0_ratios", "energy_ratios"};
        const char* output_names[] = {"Identity", "Identity_1", "Identity_2"};

        // create an array of ORT values
        std::vector<Ort::Value> input_tensors;
        // NOTE: Cannot pre-define the tensors in a separate variable, might cause pointer issues!
        input_tensors.emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size()));
        input_tensors.emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, speaker_ids.data(), speaker_ids.size(), speaker_ids_shape.data(), speaker_ids_shape.size()));
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, speed_ratios.data(), speed_ratios.size(), speed_ratios_shape.data(), speed_ratios_shape.size()));
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, f0_ratios.data(), f0_ratios.size(), f0_ratios_shape.data(), f0_ratios_shape.size()));
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, energy_ratios.data(), energy_ratios.size(), energy_ratios_shape.data(), energy_ratios_shape.size()));

        // run inference
        // infer; LightSpeech returns 3 outputs: (mel, duration, pitch)
        std::vector<Ort::Value> outputs = lightspeech.Run(run_options, input_names, input_tensors.data(), (size_t)5, output_names, (size_t)3);
        // NOTE: FastSpeech2 returns >3 outputs!

        // get durations for visemes
        int32_t* durations_tensor_ptr = outputs[1].GetTensorMutableData<int32_t>();
        // use output shape to iterate through pointer later!
        auto outputs_durations_info = outputs[1].GetTensorTypeAndShapeInfo();
        size_t total_durations_len = outputs_durations_info.GetElementCount();
        // copy durations to an int vector for visemes
        std::vector<int32_t> durations_tensor_vector;
        for (size_t i = 0; i != total_durations_len; ++i) {
          durations_tensor_vector.push_back(durations_tensor_ptr[i]);
        }

        const char* input_names_melgan[] = {"mels"};
        const char* output_names_melgan[] = {"Identity"};

        // infer melgan
        std::vector<Ort::Value> outputs_melgan = mbmelgan.Run(run_options, input_names_melgan, &outputs[0], (size_t)1, output_names_melgan, (size_t)1);
        
        // NOTE: all ONNX outputs are pointers!
        // Get pointer to output tensor float values
        float* audio_tensor_ptr = outputs_melgan[0].GetTensorMutableData<float>();
        // use output shape to iterate through pointer later!
        auto outputs_melgan_info = outputs_melgan[0].GetTensorTypeAndShapeInfo();
        size_t total_audio_len = outputs_melgan_info.GetElementCount();

        // copy floats to a float vector for export
        std::vector<float> audio_tensor_vector;
        for (size_t i = 0; i != total_audio_len; ++i) {
          audio_tensor_vector.push_back(audio_tensor_ptr[i]);
        }
    
        return audio_tensor_vector;
}
