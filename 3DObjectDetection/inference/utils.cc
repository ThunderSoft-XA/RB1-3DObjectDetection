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

#include <unistd.h> // NOLINT(build/include_order)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "log.h"
#include "utils.h"

namespace inference {

std::vector<uint8_t> decode_bmp(const uint8_t *input, int row_size, int width,
                                int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
      case 1:
        output[dst_pos] = input[src_pos];
        break;
      case 3:
        // BGR -> RGB
        output[dst_pos] = input[src_pos + 2];
        output[dst_pos + 1] = input[src_pos + 1];
        output[dst_pos + 2] = input[src_pos];
        break;
      case 4:
        // BGRA -> RGBA
        output[dst_pos] = input[src_pos + 2];
        output[dst_pos + 1] = input[src_pos + 1];
        output[dst_pos + 2] = input[src_pos];
        output[dst_pos + 3] = input[src_pos + 3];
        break;
      default:
        LOG(FATAL) << "Unexpected number of channels: " << channels;
        break;
      }
    }
  }
  return output;
}

std::vector<uint8_t> read_bmp(const std::string &input_bmp_name, int *width,
                              int *height, int *channels, Settings *s) {
  int begin, end;

  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    LOG(FATAL) << "input file " << input_bmp_name << " not found";
    exit(-1);
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

//   if (s->verbose)
    LOG(INFO) << "len: " << len;

  std::vector<uint8_t> img_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(img_bytes.data()), len);
  const int32_t header_size =
      *(reinterpret_cast<const int32_t *>(img_bytes.data() + 10));
  *width = *(reinterpret_cast<const int32_t *>(img_bytes.data() + 18));
  *height = *(reinterpret_cast<const int32_t *>(img_bytes.data() + 22));
  const int32_t bpp =
      *(reinterpret_cast<const int32_t *>(img_bytes.data() + 28));
  *channels = bpp / 8;

//   if (s->verbose)
    LOG(INFO) << "width, height, channels: " << *width << ", " << *height
              << ", " << *channels;

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * *channels * *width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);

  // Decode image, allocating tensor once the image size is known
  const uint8_t *bmp_pixels = &img_bytes[header_size];
  return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                    top_down);
}

void display_usage() {
  LOG(INFO)
      << "label_image\n"
      << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
      << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--gl_backend, -g: [0|1]: use GL GPU Delegate on Android\n"
      << "--hexagon_delegate, -j: [0|1]: use Hexagon Delegate on Android\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--image, -i: image_name.bmp\n"
      << "--labels, -l: labels for the model\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--profiling, -p: [0|1], profiling or not\n"
      << "--num_results, -r: number of results to show\n"
      << "--threads, -t: number of threads\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      << "--xnnpack_delegate, -x [0:1]: xnnpack delegate\n";
}

TfLiteStatus ReadLabelsFile(const string &file_name,
                            std::vector<string> *result,
                            size_t *found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(ERROR) << "Labels file " << file_name << " not found";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void PrintProfilingInfo(const profiling::ProfileEvent *e,
                        uint32_t subgraph_index, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symbolic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Subgraph " << std::setw(3) << std::setprecision(3)
            << subgraph_index << ", Node " << std::setw(3)
            << std::setprecision(3) << op_index << ", OpCode " << std::setw(3)
            << std::setprecision(3) << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code));
}

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

#if 0
int Main(int argc, char **argv) {
  DelegateProviders delegate_providers;
  bool parse_result = delegate_providers.InitFromCmdlineArgs(
      &argc, const_cast<const char **>(argv));
  if (!parse_result) {
    return EXIT_FAILURE;
  }

  Settings s;

  int c;
  while (true) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"image", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {"max_profiling_buffer_entries", required_argument, nullptr, 'e'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"gl_backend", required_argument, nullptr, 'g'},
        {"hexagon_delegate", required_argument, nullptr, 'j'},
        {"xnnpack_delegate", required_argument, nullptr, 'x'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:d:e:f:g:i:j:l:m:p:r:s:t:v:w:x:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c) {
    case 'a':
      s.accel = strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
      break;
    case 'b':
      s.input_mean = strtod(optarg, nullptr);
      break;
    case 'c':
      s.loop_count =
          strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
      break;
    case 'e':
      s.max_profiling_buffer_entries =
          strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
      break;
    case 'f':
      s.allow_fp16 =
          strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
      break;
    case 'g':
      s.gl_backend =
          strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
      break;
    case 'i':
      s.input_bmp_name = optarg;
      break;
    case 'j':
      s.hexagon_delegate = strtol(optarg, nullptr, 10);
      break;
    case 'l':
      s.labels_file_name = optarg;
      break;
    case 'm':
      s.model_name = optarg;
      break;
    case 'p':
      s.profiling =
          strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
      break;
    case 'r':
      s.number_of_results =
          strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
      break;
    case 's':
      s.input_std = strtod(optarg, nullptr);
      break;
    case 't':
      s.number_of_threads = strtol( // NOLINT(runtime/deprecated_fn)
          optarg, nullptr, 10);
      break;
    case 'v':
      s.verbose = strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
      break;
    case 'w':
      s.number_of_warmup_runs =
          strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
      break;
    case 'x':
      s.xnnpack_delegate =
          strtol(optarg, nullptr, 10); // NOLINT(runtime/deprecated_fn)
      break;
    case 'h':
    case '?':
      /* getopt_long already printed an error message. */
      display_usage();
      exit(-1);
    default:
      exit(-1);
    }
  }

  delegate_providers.MergeSettingsIntoParams(s);
  //   RunInference(&s, delegate_providers);
  return 0;
}

#endif

// NHWC source data is uint8_t to resize convert float or other 
std::vector<uint8_t> mat2vector(cv::Mat img, cv::Size2d size = {512,512}) {
	cv::resize(img, img, size);
	// cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	// img.convertTo(img, CV_32FC3);
	//数据归一化
	// img = img / 255.0;
	//将rgb数据分离为单通道
	std::vector<cv::Mat> mv;
	cv::split(img, mv);
	std::vector<uint8_t> B = mv[0].reshape(1, 1);
	std::vector<uint8_t> G = mv[1].reshape(1, 1);
	std::vector<uint8_t> R = mv[2].reshape(1, 1);
	//RGB数据合并
	std::vector<uint8_t> input_data;
    for(int i = 0; i < img.cols * img.rows; i++) {
        input_data.push_back(R[i]);
        input_data.push_back(G[i]);
        input_data.push_back(B[i]);
    }
	// input_data.insert(input_data.end(), R.begin(), R.end());
	// input_data.insert(input_data.end(), G.begin(), G.end());
	// input_data.insert(input_data.end(), B.begin(), B.end());
	return input_data;
}


} // namespace inference
