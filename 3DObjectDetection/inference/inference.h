#ifndef INFERENCE_H_
#define INFERENCE_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace inference {

typedef enum {
  LOAD_MODEL_FAILED = -4,
  SET_PARAM_FAILED,
  LOAD_DATA_FAILED,
  INFERENCE_FAILED,
  SUCCESS = 0
} InferenceERROR;

class Inference {
public:
  virtual InferenceERROR loadModel() = 0;
  virtual InferenceERROR setInferenceParam() = 0;
  InferenceERROR loadData();
  InferenceERROR inferenceModel();
};

} // namespace inference

#endif // INFERENCE_H_
