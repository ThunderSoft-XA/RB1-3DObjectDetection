/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef UTILS_H_
#define UTILS_H_

#include <cstdlib>
#include <iomanip>

#include "delegate_tf.h"
#include "settings.h"
#include "utils_impl.h"

namespace inference {

using namespace tf;

std::vector<uint8_t> read_bmp(const std::string &input_bmp_name, int *width,
                              int *height, int *channels, Settings *s);

template <class T>
void resize(T *out, uint8_t *in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, Settings *s);

// explicit instantiation
template void resize<float>(float *, unsigned char *, int, int, int, int, int,
                            int, Settings *);
template void resize<int8_t>(int8_t *, unsigned char *, int, int, int, int, int,
                             int, Settings *);
template void resize<uint8_t>(uint8_t *, unsigned char *, int, int, int, int,
                              int, int, Settings *);

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string &file_name,
                            std::vector<string> *result,
                            size_t *found_label_count);

void PrintProfilingInfo(const profiling::ProfileEvent *e,
                        uint32_t subgraph_index, uint32_t op_index,
                        TfLiteRegistration registration);

double get_us(struct timeval t);

void display_usage();

int Main(int argc, char **argv);

// template <class T> std::vector<T> mat2Vector(const cv::Mat &mat);

// template std::vector<uint8_t> mat2Vector<uint8_t>(const cv::Mat &mat);

std::vector<uint8_t> mat2vector(cv::Mat img, cv::Size2d size);

} // namespace inference

#endif // UTILS_H_
