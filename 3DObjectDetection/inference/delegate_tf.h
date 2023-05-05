#ifndef DELEGATE_TF_H_
#define DELEGATE_TF_H_

#include <fcntl.h>     // NOLINT(build/include_order)
#include <getopt.h>    // NOLINT(build/include_order)
#include <sys/time.h>  // NOLINT(build/include_order)
#include <sys/types.h> // NOLINT(build/include_order)
#include <sys/uio.h>   // NOLINT(build/include_order)
#include <unistd.h>    // NOLINT(build/include_order)

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

#include "absl/memory/memory.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#include "inference.h"
#include "log.h"
#include "settings.h"

namespace inference {
namespace tf {

using namespace tflite;

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;

class DelegateProviders {
public:
  DelegateProviders() : delegate_list_util_(&params_) {
    delegate_list_util_.AddAllDelegateParams();
  }

  // Initialize delegate-related parameters from parsing command line arguments,
  // and remove the matching arguments from (*argc, argv). Returns true if all
  // recognized arg values are parsed correctly.
  bool InitFromCmdlineArgs(int *argc, const char **argv) {
    std::vector<tflite::Flag> flags;
    delegate_list_util_.AppendCmdlineFlags(&flags);

    const bool parse_result = Flags::Parse(argc, argv, flags);
    if (!parse_result) {
      std::string usage = Flags::Usage(argv[0], flags);
      LOG(ERROR) << usage;
    }
    return parse_result;
  }

  // According to passed-in settings `s`, this function sets corresponding
  // parameters that are defined by various delegate execution providers. See
  // lite/tools/delegates/README.md for the full list of parameters defined.
  void MergeSettingsIntoParams(const Settings &s) {
    // Parse settings related to GPU delegate.
    // Note that GPU delegate does support OpenCL. 'gl_backend' was introduced
    // when the GPU delegate only supports OpenGL. Therefore, we consider
    // setting 'gl_backend' to true means using the GPU delegate.
    if (s.gl_backend) {
      if (!params_.HasParam("use_gpu")) {
        LOG(WARN) << "GPU deleate execution provider isn't linked or GPU "
                     "delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_gpu", true);
        // The parameter "gpu_inference_for_sustained_speed" isn't available for
        // iOS devices.
        if (params_.HasParam("gpu_inference_for_sustained_speed")) {
          params_.Set<bool>("gpu_inference_for_sustained_speed", true);
        }
        params_.Set<bool>("gpu_precision_loss_allowed", s.allow_fp16);
      }
    }

    // Parse settings related to NNAPI delegate.
    if (s.accel) {
      if (!params_.HasParam("use_nnapi")) {
        LOG(WARN) << "NNAPI deleate execution provider isn't linked or NNAPI "
                     "delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_nnapi", true);
        params_.Set<bool>("nnapi_allow_fp16", s.allow_fp16);
      }
    }

    // Parse settings related to Hexagon delegate.
    if (s.hexagon_delegate) {
      if (!params_.HasParam("use_hexagon")) {
        LOG(WARN) << "Hexagon deleate execution provider isn't linked or "
                     "Hexagon delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_hexagon", true);
        params_.Set<bool>("hexagon_profiling", s.profiling);
      }
    }

    // Parse settings related to XNNPACK delegate.
    if (s.xnnpack_delegate) {
      if (!params_.HasParam("use_xnnpack")) {
        LOG(WARN) << "XNNPACK deleate execution provider isn't linked or "
                     "XNNPACK delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_xnnpack", true);
        params_.Set<bool>("num_threads", s.number_of_threads);
      }
    }
  }

  // Create a list of TfLite delegates based on what have been initialized (i.e.
  // 'params_').
  std::vector<ProvidedDelegateList::ProvidedDelegate>
  CreateAllDelegates() const {
    return delegate_list_util_.CreateAllRankedDelegates();
  }

private:
  // Contain delegate-related parameters that are initialized from command-line
  // flags.
  tflite::tools::ToolParams params_;

  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;
};

} // namespace tf

} // namespace inference

#endif // DELEGATE_TF_H_