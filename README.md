# YOLOX-YOLOV5-YOLOV8-OpenVINO in C++

This tutorial includes a C++ inference of YOLOX/YOLOV5/YOLOV8 for OpenVINO.

## Install OpenVINO Toolkit

Please visit [Openvino Homepage](https://docs.openvinotoolkit.org/latest/get_started_guides.html) for more details.

## Environment
OpenVINO_2022.3.0      
OpenCV-4.6.0

## Convert model

1. Export ONNX model
   
2. Convert ONNX to OpenVINO 

   Install requirements for convert tool

   ```shell
   pip install openvino-dev
   ```

   Convert ONNX into the OpenVINO IR

   FP32
   ```shell
   mo -m <onnx_model_path>  --output_dir <MODEL_DIR> 
   ```
   FP16
   ```shell
   mo -m <onnx_model_path>  --output_dir <MODEL_DIR> --compress_to_fp16
   ```

   INT8 Quantization with POT
   ```shell
   pot -q default -m <ir_model_xml> -w <ir_model_bin> --engine simplified --data-source <data_dir> --output-dir <output_dir_name> --direct-dump --name <int8_model_name>
   ```

   visit [Openvino POT](https://docs.openvino.ai/latest/notebooks/114-quantization-simplified-mode-with-output.html) for more details.

## Build 

### Linux
```shell
mkdir build
cd build
cmake ..
make
```

## Demo

### c++

```shell
./detect  <...><...> 
```