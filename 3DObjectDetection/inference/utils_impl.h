/**
 * @file utils_imp.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-02-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef UTILS_IMPL_H_
#define UTILS_IMPL_H_

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

#include "inference_tf.hpp"
#include "utils.h"

namespace inference {

using namespace tflite;

template <class T>
void resize(T *out, uint8_t *in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, tf::Settings *s) {
  int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<Interpreter> interpreter(new Interpreter);

  int base_index = 0;

  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);

  ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration *resize_op =
      resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR, 1);
  auto *params = reinterpret_cast<TfLiteResizeBilinearParams *>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  params->half_pixel_centers = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                     nullptr);

  interpreter->AllocateTensors();

  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }

  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;

  interpreter->Invoke();

  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;
  std::cout << "std  input mean: " << s->input_mean << std::endl; 
  for (int i = 0; i < output_number_of_pixels; i++) {
    switch (s->input_type) {
    case kTfLiteFloat32:
      out[i] = (output[i] - s->input_mean) / s->input_std;
      break;
    case kTfLiteInt8:
      out[i] = static_cast<int8_t>(output[i] - 128);
      break;
    case kTfLiteUInt8:
      out[i] = static_cast<uint8_t>(output[i]);
      break;
    default:
      break;
    }
    
    if(i < 100 ) {
        std::cout << out[i] << " ";
        if(i % 10 == 0){
            std::cout << std::endl;
    }
    }
  }
}

// template <class T> std::vector<T> mat2Vector(const cv::Mat &mat) {
//   return (std::vector<T>)(mat.clone().reshape(1, 1)); //通道数不变，按行转为一行
// }

} // namespace inference

#endif // UTILS_IMPL_H_
