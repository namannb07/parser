from pathlib import Path
import zipfile, textwrap

base = Path("/mnt/data/rfdetr_deepstream_parser")
base.mkdir(exist_ok=True)

cpp = r'''#include "rfdetr_parser.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kClassId = 0;
constexpr int kBackgroundClassIndex = -1; // Last logit slot for contiguous 0-based custom datasets.
constexpr float kFallbackThreshold = 0.25f;
constexpr float kEpsilon = 1.0e-6f;

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

size_t volume(const NvDsInferDims& dims) {
    if (dims.numDims <= 0) return 0;
    size_t n = 1;
    for (int i = 0; i < dims.numDims; ++i) {
        if (dims.d[i] <= 0) return 0;
        n *= static_cast<size_t>(dims.d[i]);
    }
    return n;
}

int lastDim(const NvDsInferDims& dims) {
    return dims.numDims > 0 ? dims.d[dims.numDims - 1] : 0;
}

float sigmoid(float x) {
    // Numerically stable sigmoid.
    if (x >= 0.0f) {
        const float z = std::exp(-std::min(x, 88.0f));
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(std::max(x, -88.0f));
    return z / (1.0f + z);
}

bool isFiniteBox(float cx, float cy, float w, float h) {
    return std::isfinite(cx) && std::isfinite(cy) &&
           std::isfinite(w) && std::isfinite(h) &&
           w > 0.0f && h > 0.0f;
}

} // namespace

extern "C" bool NvDsInferParseCustomRFDETR(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    const NvDsInferLayerInfo* boxesLayer = nullptr;  // RF-DETR output "dets"
    const NvDsInferLayerInfo* logitsLayer = nullptr; // RF-DETR output "labels"

    // Match by binding name, never by shape alone. RF-DETR warns that shapes can
    // be ambiguous when the number of classes is three.
    for (const auto& layer : outputLayersInfo) {
        const std::string name = lower(layer.layerName);
        if (name.find("dets") != std::string::npos) {
            boxesLayer = &layer;
        } else if (name.find("labels") != std::string::npos) {
            logitsLayer = &layer;
        }
    }

    if (!boxesLayer || !logitsLayer) {
        std::cerr << "[RF-DETR parser] Required outputs not found. Expected names "
                  << "'dets' and 'labels'. Actual outputs:";
        for (const auto& layer : outputLayersInfo) {
            std::cerr << " " << (layer.layerName ? layer.layerName : "<unnamed>");
        }
        std::cerr << std::endl;
        return false;
    }

    if (!boxesLayer->buffer || !logitsLayer->buffer) {
        std::cerr << "[RF-DETR parser] Output buffer is null." << std::endl;
        return false;
    }

    // This implementation expects FP32 output buffers, which is the normal
    // DeepStream nvinfer parser representation for FP32 RF-DETR output tensors.
    // Verify the engine/output data type if using a custom FP16 output binding.
    const int boxWidth = lastDim(boxesLayer->inferDims);
    const int logitWidth = lastDim(logitsLayer->inferDims);
    const size_t boxElements = volume(boxesLayer->inferDims);
    const size_t logitElements = volume(logitsLayer->inferDims);

    if (boxWidth != 4 || logitWidth < 2 ||
        boxElements == 0 || logitElements == 0 ||
        boxElements % 4 != 0 || logitElements % static_cast<size_t>(logitWidth) != 0) {
        std::cerr << "[RF-DETR parser] Unexpected tensor dimensions. dets elements="
                  << boxElements << " last_dim=" << boxWidth
                  << ", labels elements=" << logitElements
                  << " last_dim=" << logitWidth << std::endl;
        return false;
    }

    const size_t numQueries = boxElements / 4;
    const size_t logitQueries = logitElements / static_cast<size_t>(logitWidth);
    if (numQueries != logitQueries) {
        std::cerr << "[RF-DETR parser] Query count mismatch: boxes=" << numQueries
                  << ", logits=" << logitQueries << std::endl;
        return false;
    }

    const float* boxes = static_cast<const float*>(boxesLayer->buffer);
    const float* logits = static_cast<const float*>(logitsLayer->buffer);

    // One-class custom RF-DETR checkpoint trained with contiguous category ID 0:
    // the last logit slot is background, and slot 0 is the foreground class.
    int backgroundIndex = kBackgroundClassIndex;
    if (backgroundIndex < 0) backgroundIndex += logitWidth;
    if (backgroundIndex < 0 || backgroundIndex >= logitWidth) {
        std::cerr << "[RF-DETR parser] Invalid background index." << std::endl;
        return false;
    }

    int foregroundIndex = 0;
    if (foregroundIndex == backgroundIndex) {
        std::cerr << "[RF-DETR parser] No foreground logit slot." << std::endl;
        return false;
    }

    float threshold = kFallbackThreshold;
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
        threshold = detectionParams.perClassPreclusterThreshold[0];
    }
    threshold = std::max(0.0f, std::min(threshold, 1.0f));

    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    for (size_t q = 0; q < numQueries; ++q) {
        const float score = sigmoid(logits[q * logitWidth + foregroundIndex]);
        if (!std::isfinite(score) || score < threshold) {
            continue;
        }

        // RF-DETR exports normalized center-x, center-y, width, height.
        const float cx = boxes[q * 4 + 0];
        const float cy = boxes[q * 4 + 1];
        const float bw = boxes[q * 4 + 2];
        const float bh = boxes[q * 4 + 3];

        if (!isFiniteBox(cx, cy, bw, bh)) {
            continue;
        }

        float left   = (cx - 0.5f * bw) * netW;
        float top    = (cy - 0.5f * bh) * netH;
        float right  = (cx + 0.5f * bw) * netW;
        float bottom = (cy + 0.5f * bh) * netH;

        // Clip to network input bounds.
        left   = std::max(0.0f, std::min(left, netW));
        top    = std::max(0.0f, std::min(top, netH));
        right  = std::max(0.0f, std::min(right, netW));
        bottom = std::max(0.0f, std::min(bottom, netH));

        const float width = right - left;
        const float height = bottom - top;
        if (width <= kEpsilon || height <= kEpsilon) {
            continue;
        }

        NvDsInferObjectDetectionInfo obj{};
        obj.classId = kClassId;
        obj.left = left;
        obj.top = top;
        obj.width = width;
        obj.height = height;
        obj.detectionConfidence = score;
        objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomRFDETR);
'''

header = r'''#ifndef RFDETR_PARSER_H
#define RFDETR_PARSER_H

#include "nvdsinfer_custom_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

bool NvDsInferParseCustomRFDETR(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

#ifdef __cplusplus
}
#endif

#endif
'''

makefile = r'''# Adjust these paths if DeepStream is installed elsewhere.
DS_ROOT ?= /opt/nvidia/deepstream/deepstream
CUDA_HOME ?= /usr/local/cuda
CXX ?= g++

TARGET := librfdetr_parser.so
SRC := rfdetr_parser.cpp

CXXFLAGS := -O3 -std=c++14 -fPIC -Wall -Wextra \
            -I$(DS_ROOT)/sources/includes \
            -I$(CUDA_HOME)/include

LDFLAGS := -shared

all: $(TARGET)

$(TARGET): $(SRC) rfdetr_parser.h
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
'''

config = r'''# RF-DETR one-class detector - DeepStream 7.1
[property]
gpu-id=0
onnx-file=/path/to/rfdetr.onnx
model-engine-file=/path/to/rfdetr_b1_fp16.engine
labelfile-path=/path/to/labels.txt

batch-size=1
network-type=0
process-mode=1
num-detected-classes=1

# RF-DETR custom parser
custom-lib-path=/path/to/librfdetr_parser.so
parse-bbox-func-name=NvDsInferParseCustomRFDETR
parse-func=0

# Use the same input dimensions used when exporting the model.
# This example assumes RF-DETR Nano export is 576 x 576.
infer-dims=3;576;576
maintain-aspect-ratio=0
symmetric-padding=0

# RF-DETR ONNX graph expects RGB values normalized using ImageNet mean/std.
# Configure nvinfer preprocessing to match the ONNX export and your DeepStream
# version's supported input-tensor settings. Do not assume scale=1/255 alone
# is sufficient: verify preprocessing separately.
net-scale-factor=1.0
model-color-format=0

# The parser performs sigmoid and normalized cxcywh -> pixel xywh conversion.
# DETR-style query predictions do not require conventional NMS.
cluster-mode=4
pre-cluster-threshold=0.25

# Optional debugging
# output-tensor-meta=1
'''

readme = r'''# RF-DETR DeepStream C++ parser

Target: Jetson Orin Nano Super, DeepStream 7.1, TensorRT 10.3.

## Assumed RF-DETR export contract

The parser is written for the RF-DETR raw detection ONNX outputs documented by Roboflow:

- `dets`: normalized boxes in `cx, cy, w, h` order
- `labels`: raw, unactivated class logits
- one-class custom checkpoint with contiguous class ID 0
- final logit slot is background
- FP32 output buffers exposed to the DeepStream custom parser

The parser matches outputs by binding name, applies sigmoid to class-0 logits,
converts normalized boxes to network-input pixel coordinates, clips them, and
emits class ID 0. It intentionally does not run NMS: DETR uses set predictions.

**Important:** This is a parser implementation against that documented output
contract, not a claim that every RF-DETR export has identical binding names,
precision, or class-slot mapping. Check your actual ONNX/engine output metadata
before deployment. If your checkpoint was trained with sparse COCO category IDs,
the background-slot assumption must be changed.

## Build

```bash
cd rfdetr_deepstream_parser
make
```

Expected output: `librfdetr_parser.so`.

If DeepStream is installed in a different directory:

```bash
make DS_ROOT=/opt/nvidia/deepstream/deepstream
```

## Configure nvinfer

Copy the relevant properties from `config_infer_rfdetr.txt` into your PGIE
configuration. Set the real ONNX/engine and parser library paths.

Use `cluster-mode=4` (no clustering) for DETR-style predictions. Set
`pre-cluster-threshold` to the desired confidence threshold.

## Verify tensor names and shapes

Inspect the ONNX graph before building the TensorRT engine:

```bash
python3 - <<'PY'
import onnx
m = onnx.load("rfdetr.onnx")
for o in m.graph.output:
    dims = []
    for d in o.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else d.dim_param)
    print(o.name, dims)
PY
```

Expected logical outputs are typically `dets` with final dimension 4 and
`labels` with final dimension 2 for a one-class model. Binding names may have
TensorRT/DeepStream prefixes or suffixes; the parser currently matches names
containing `dets` and `labels`.

If output buffers are FP16 rather than FP32, this parser must be adapted to
read half values correctly; do not reinterpret FP16 buffers as float.

## Preprocessing warning

RF-DETR's documented ONNX example uses RGB, NCHW, ImageNet normalization:
mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225].
The parser does not preprocess images. Ensure DeepStream's input preprocessing
matches the exact export contract. A scale factor of 1/255 alone is not
equivalent to ImageNet normalization.

## Known limitations

- Assumes one foreground class at logit index 0 and background at final index.
- Assumes FP32 output buffers.
- Assumes output names contain `dets` and `labels`.
- Assumes normalized `cxcywh` boxes.
- Does not perform NMS.
- Coordinates are returned in network-input pixel space. DeepStream handles
  mapping to the source frame using its normal inference metadata path.
'''

files = {
    "rfdetr_parser.cpp": cpp,
    "rfdetr_parser.h": header,
    "Makefile": makefile,
    "config_infer_rfdetr.txt": config,
    "README.md": readme,
}
for name, content in files.items():
    (base / name).write_text(content)

zip_path = Path("/mnt/data/rfdetr_deepstream_parser.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(base / name, arcname=f"rfdetr_deepstream_parser/{name}")

print(f"Created: {zip_path}")
print("Included:", ", ".join(files.keys()))
