#ifndef INFERENCE_TF_H_
#define INFERENCE_TF_H_

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "absl/memory/memory.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#include <tensorflow/lite/delegates/gpu/cl/gpu_api_delegate.h>
#include <tensorflow/lite/delegates/gpu/common/model_builder.h>
#include <tensorflow/lite/delegates/gpu/common/status.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>

#include <tensorflow/lite/delegates/hexagon/hexagon_delegate.h>

#include "inference.h"
#include "settings.h"
#include "utils.h"

namespace inference {

extern double get_us(struct timeval t);
extern std::vector<uint8_t> mat2vector(cv::Mat img, cv::Size2d size);
namespace tf {

using InputPair = std::pair<int,cv::Mat>;
using InputPairVec = std::vector<InputPair>;

typedef enum {
    NONE = 0,
    GPU,
    HEXAGON,
    XNNPACK
} DelegateType;

class TFInference : public Inference {

public:
  InferenceERROR loadModel();
  InferenceERROR setInferenceParam();
  void setDelegate();

  int getInputsNum() {
    assert(interpreter != NULL);
    return interpreter->inputs().size();
  }

  InferenceERROR loadData(InputPairVec _data_mat_mat);

  void setSettings(Settings *_settings) {
    this->settings = _settings;
  }

  auto getResultType() {
    assert(interpreter != NULL);
    return interpreter->tensor(interpreter->outputs()[0])->type;
  }
  template <class T> InferenceERROR inferenceModel(std::vector<T> &_result_vec);

private:
  std::string model_path;
  Settings *settings;

  DelegateType delegate_type = HEXAGON;

  TfLiteDelegate *delegate;
  DelegateProviders delegate_providers;

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
};


template <class T>
InferenceERROR TFInference::inferenceModel(std::vector<T> &_result_vec) {
  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);
  for (int i = 0; i < settings->loop_count; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(ERROR) << "Failed to invoke tflite!";
      return INFERENCE_FAILED;
    }
  }
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "invoked";
  LOG(INFO) << "average time: "
            << (get_us(stop_time) - get_us(start_time)) /
                   (settings->loop_count * 1000)
            << " ms";

  std::vector<std::pair<float, int>> top_results;

  int output = interpreter->outputs()[0];
  TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = (output_dims->data[0] == 0 ? 1 : output_dims->data[0]) 
        * (output_dims->data[1] == 0 ? 1 : output_dims->data[1]) 
        * (output_dims->data[2] == 0 ? 1 : output_dims->data[2])
        * (output_dims->data[3] == 0 ? 1 : output_dims->data[3]);

  LOG(INFO) << "output_size: " << output_size
            << output_dims->data[0] << "x"
            << output_dims->data[1] << "x"
            << output_dims->data[2] << "x"
            << output_dims->data[3];
  float value = 0.0;
  // interpreter->typed_output_tensor<uint8_t>(0)
  for (int i = 0; i < output_size; ++i) {
    switch (interpreter->tensor(output)->type) {
    case kTfLiteFloat32:
      value = interpreter->typed_output_tensor<float>(0)[i];
      break;
    case kTfLiteInt8:
      value = (interpreter->typed_output_tensor<int8_t>(0)[i] + 128) / 256.0;
      break;
    case kTfLiteUInt8:
      value = interpreter->typed_output_tensor<uint8_t>(0)[i] / 255.0;
      break;
    default:
      LOG(ERROR) << "cannot handle output type "
                 << interpreter->tensor(output)->type << " yet";
      exit(-1);
    }
    _result_vec.push_back(value);
  }
    // int count = 0;
    // for(auto value : _result_vec) {
    //     if (count > 100) {
    //         break;
    //     }
    //     std::cout << value << " ";
    //     count++;
    //     if (count % 10 == 0) {
    //         std::cout << std::endl;
    //     }

    // }
  std::cout << "_result_vec size : " << _result_vec.size() << std::endl;
  return SUCCESS;
}

} // namespace tf
} // namespace inference

#endif // INFERENCE_TF_H_
