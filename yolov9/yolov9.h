#pragma once
#include<string>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<openvino/openvino.hpp>
#include <fstream>
#include <vector>
#include <random>

struct Config {
	float confThreshold;
	float nmsThreshold;
	float scoreThreshold;
	int inpWidth;
	int inpHeight;
	std::string onnx_path;
};

struct ImageShape
{
	float dw;
	float dh;
    float width;
    float height;
};

struct Detection {
	int class_id;
	float confidence;
	cv::Rect box;
};

class YOLOV9{
public:
	YOLOV9(Config config);
	~YOLOV9();
	void detect(cv::Mat& frame);

private:
	float confThreshold;
	float nmsThreshold;
	float scoreThreshold;
	int inpWidth;
	int inpHeight;
    float scale;
	ImageShape imageshape;
	std::string onnx_path;
	ov::Tensor input_tensor;
	ov::InferRequest infer_request;
	ov::CompiledModel compiled_model;
	void initialmodel();
	void preprocess_img(cv::Mat& frame);
	void postprocess_img(cv::Mat& frame, float* detections, ov::Shape & output_shape);
};