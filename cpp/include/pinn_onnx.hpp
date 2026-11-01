#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <array>
#include <iostream>

namespace pinn {

class PinnOnnx {
public:
    PinnOnnx(const std::string& model_path) {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "pinn");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
    }

    float predict(float tau, float S) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // PyTorch ha esportato 2 input separati: tau e S (shape: [1, 1])
        std::array<float, 1> tau_data = {tau};
        std::array<float, 1> S_data = {S};
        std::array<int64_t, 2> shape = {1, 1};

        Ort::Value tau_tensor = Ort::Value::CreateTensor<float>(
            memory_info, tau_data.data(), tau_data.size(), shape.data(), shape.size());
        Ort::Value S_tensor = Ort::Value::CreateTensor<float>(
            memory_info, S_data.data(), S_data.size(), shape.data(), shape.size());

        std::array<Ort::Value, 2> input_tensors = {std::move(tau_tensor), std::move(S_tensor)};
        const char* input_names[] = {"tau", "S"};
        const char* output_names[] = {"output"}; // Se dà errore, potrebbe essere "mul_24"

        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, 
            input_names, input_tensors.data(), input_tensors.size(), 
            output_names, 1
        );

        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        return floatarr[0];
    }

private:
    Ort::Env env_{nullptr};
    Ort::Session session_{nullptr};
};

} // namespace pinn