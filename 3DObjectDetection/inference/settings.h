/**
 * @file settings.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-02-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef SETTINGS_H_
#define SETTINGS_H_

#include <iostream>
#include <string>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace inference {
namespace tf {

struct Settings {
//   bool verbose = false;
  bool accel = false;
  TfLiteType input_type = kTfLiteFloat32;
  bool profiling = false;
  bool allow_fp16 = false;
  bool gl_backend = false;
  bool hexagon_delegate = false;
  bool xnnpack_delegate = false;
  int loop_count = 1;
  float input_mean = 0;
  float input_std = 1;
  std::string model_name = "./magenta_arbitrary-image-stylization-v1-256_int8_prediction_1.tflite";
//   tflite::FlatBufferModel *model;
  std::string input_bmp_name = "./grace_hopper.bmp";
//   std::string labels_file_name = "./labels.txt";
  int number_of_threads = 1;
//   int number_of_results = 5;
//   int max_profiling_buffer_entries = 1024;
//   int number_of_warmup_runs = 2;
};

} // namespace tf
} // namespace inference

#endif // SETTINGS_H_
