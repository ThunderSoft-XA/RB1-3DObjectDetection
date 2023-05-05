/**
 * @file inference.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-02-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <fcntl.h>     // NOLINT(build/include_order)
#include <getopt.h>    // NOLINT(build/include_order)
#include <sys/time.h>  // NOLINT(build/include_order)
#include <sys/types.h> // NOLINT(build/include_order)
#include <sys/uio.h>   // NOLINT(build/include_order)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "inference_tf.hpp"
#include "log.h"

namespace inference {
namespace tf {

InferenceERROR TFInference::loadModel() {

  if (!settings->model_name.c_str()) {
    LOG(ERROR) << "no model file name";
    exit(-1);
  }

  model = tflite::FlatBufferModel::BuildFromFile(settings->model_name.c_str());
  if (!model) {
    LOG(ERROR) << "Failed to mmap model " << settings->model_name;
    exit(-1);
  }
//   settings->model = model.get();
  LOG(INFO) << "Loaded model " << settings->model_name;
  model->error_reporter();
  LOG(INFO) << "resolved reporter";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(ERROR) << "Failed to construct interpreter";
    return LOAD_MODEL_FAILED;
  }
}

void TFInference::setDelegate() {
    if (this->delegate_type == DelegateType::GPU) {
        TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
        options.is_precision_loss_allowed = true;
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
        options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
        options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        options.max_delegated_partitions = 1;

        auto* delegate = TfLiteGpuDelegateV2Create(&options);
    
        auto status = interpreter->ModifyGraphWithDelegate(delegate);
        // auto status = _interpreter->AllocateTensors();
        if (status != kTfLiteOk) {
            LOG(ERROR) << "Failed to allocate the memory for tensors.";
            exit(-1);
        } else {
            LOG(INFO) << "\nMemory allocated for tensors.\n";
        }
    } else if(this->delegate_type == DelegateType::HEXAGON) {
        // Assuming shared libraries are under "/data/local/tmp/"
        // If files are packaged with native lib in android App then it
        // will typically be equivalent to the path provided by
        // "getContext().getApplicationInfo().nativeLibraryDir"
        std::string library_directory_path = "/data/local/tmp/";
        TfLiteHexagonInitWithPath(library_directory_path.c_str());  // Needed once at startup.
        TfLiteHexagonDelegateOptions params = {0};
        // 'delegate_ptr' Need to outlive the interpreter. For example,
        // If use case will need to resize input or anything that can trigger
        // re-applying delegates then 'delegate_ptr' need to outlive the interpreter.
        auto* delegate_ptr = TfLiteHexagonDelegateCreate(&params);
        Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
        [](TfLiteDelegate* delegate) {
            TfLiteHexagonDelegateDelete(delegate);
        });
        interpreter->ModifyGraphWithDelegate(delegate.get());
        // After usage of delegate.
        TfLiteHexagonTearDown();  // Needed once at end of app/DSP usage.
    }
}


InferenceERROR TFInference::setInferenceParam() {
  delegate_providers.MergeSettingsIntoParams(*settings);
  interpreter->SetAllowFp16PrecisionForFp32(settings->allow_fp16);

//   if (settings->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size();
    LOG(INFO) << "nodes size: " << interpreter->nodes_size();
    LOG(INFO) << "inputs: " << interpreter->inputs().size();
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0);

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point;
    }
//   }

  if (settings->number_of_threads != -1) {
    interpreter->SetNumThreads(settings->number_of_threads);
  }

//   auto delegates = delegate_providers.CreateAllDelegates();
//   for (auto &delegate : delegates) {
//     const auto delegate_name = delegate.provider->GetName();
//     if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) !=
//         kTfLiteOk) {
//       LOG(ERROR) << "Failed to apply " << delegate_name << " delegate.";
//       exit(-1);
//     } else {
//       LOG(INFO) << "Applied " << delegate_name << " delegate.";
//     }
//   }

  setDelegate();
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Failed to allocate tensors!";
    return SET_PARAM_FAILED;
  }
}

InferenceERROR TFInference::loadData(InputPairVec _mat_pair_vec) {
    assert(_mat_pair_vec.empty() != true);
    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();

    LOG(INFO) << "number of inputs: " << inputs.size();
    LOG(INFO) << "number of outputs: " << outputs.size();
    for(auto _mat_pair : _mat_pair_vec) {
        int image_width = _mat_pair.second.rows;  // 256
        int image_height = _mat_pair.second.cols; // 256
        int image_channels = _mat_pair.second.channels();

        int input = interpreter->inputs()[_mat_pair.first];
        LOG(INFO) << "input: " << input;

        // get input dimension from the input tensor metadata
        // assuming one input only
        TfLiteIntArray *dims = interpreter->tensor(input)->dims;
        int wanted_height = dims->data[1];
        int wanted_width = dims->data[2];
        int wanted_channels = dims->data[3];

        LOG(INFO) << "image_width : " << image_width
            << ", image_height : " <<  image_height;

        LOG(INFO) << "input tensor name: " << interpreter->tensor(input)->name;
        LOG(INFO) << "wanted size:" << wanted_height << " x " 
                << wanted_width << " x " << wanted_channels;

        //std::vector<uint8_t> input_data = mat2Vector<uint8_t>(_mat_pair.second);
        cv::Size2d wanted_size(wanted_width,wanted_height);
        std::vector<float> input_data;// = mat2vector(_mat_pair.second,mat_size);
        if(_mat_pair.second.channels() == 3) {
            std::cout << "mat channels is 3" << std::endl;
            _mat_pair.second.convertTo(_mat_pair.second, CV_32F);
            cv::normalize(_mat_pair.second,_mat_pair.second,1.0, 0, cv::NORM_MINMAX);
            if(image_width != wanted_width || image_height != wanted_height) {
                cv::resize(_mat_pair.second, _mat_pair.second, wanted_size);
            }
            std::vector<cv::Mat> mv;
            cv::split(_mat_pair.second, mv);
            std::vector<float> B = mv[0].reshape(1, 1);
            std::vector<float> G = mv[1].reshape(1, 1);
            std::vector<float> R = mv[2].reshape(1, 1);
            //RGB数据合并
            for(int i = 0; i < wanted_height * wanted_width; i++) {
                input_data.push_back(R[i]);
                input_data.push_back(G[i]);
                input_data.push_back(B[i]);
                // std::cout << R[i] << " " << G[i] << " " << B[i] << std::endl;
            }
            // input_data = mat2vector(_mat_pair.second,mat_size);
        } else {
            input_data = _mat_pair.second.reshape(1, 1);
        }

        std::cout << "mat convert vector finished !!! size ：" << input_data.size() << std::endl;

        settings->input_type = interpreter->tensor(input)->type;
        LOG(INFO) << "input type: " << ((settings->input_type == kTfLiteFloat32)
            ? "float32" : ((settings->input_type == kTfLiteInt8)
            ? "int8" : ((settings->input_type == kTfLiteUInt8)
            ? "uint8" : "error")));

        // skip predict input`s resize， interpreter use typed_input_tensor
        // interpreter->typed_input_tensor<float>(input)[0] = 0.1;
        LOG(INFO) << " load predict input data";
        memcpy(interpreter->typed_tensor<float>(input), &((float*)input_data.data())[0], input_data.size()*sizeof(float));
        // for(int index = 0; index < input_data.size(); index++) {
        //     LOG(INFO) << index;
        //     interpreter->typed_tensor<float>(input)[index] = (float)input_data[index];
        // }
    }
    LOG(INFO) << "load data finished ！！";
    return LOAD_DATA_FAILED;
}

} // namespace tf
} // namespace inference
