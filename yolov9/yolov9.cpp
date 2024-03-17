#include"yolov9.h"
#include<iostream>
#include<string>
#include<time.h>

using namespace cv;
using namespace std;
using namespace dnn;


const char* coconame[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

YOLOV9::YOLOV9(Config config)
{
    this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
    this->scoreThreshold = config.scoreThreshold;
	this->inpWidth = config.inpWidth;
	this->inpHeight = config.inpHeight;
	this->onnx_path = config.onnx_path;
    this->initialmodel();
}

YOLOV9::~YOLOV9(){}

void YOLOV9::detect(Mat & frame) {
    preprocess_img(frame);
    infer_request.infer();
    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    float* detections = output_tensor.data<float>();
    this->postprocess_img(frame, detections, output_shape);
}


void YOLOV9::initialmodel() {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(this->onnx_path);
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::RGB);
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255, 255, 255 });// .scale({ 112, 112, 112 });
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    this->compiled_model = core.compile_model(model, "CPU");
    this->infer_request = compiled_model.create_infer_request();
}

void YOLOV9::preprocess_img(cv::Mat& frame)
{
    try {
        float width = frame.cols;
        float height = frame.rows;
        imageshape.width = width;
        imageshape.height = height;

        cv::Size new_shape = cv::Size(inpWidth, inpHeight);
        scale = float(new_shape.width / max(width, height));
        int new_unpadW = int(round(width * scale));
        int new_unpadH = int(round(height * scale));

        int w=width,h=height;
        if(w>h)
        {
            w = inpWidth;
            h = h * scale;
        }
        else
        {
            h = inpHeight;
            w = w * scale;
        }
        cv::Mat input;
        cv::resize(frame, input, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);
        int dw = inpWidth - w;
        int dh = inpHeight - h;
        dw = dw / 2;
        dh = dh / 2;
        imageshape.dw = dw;
        imageshape.dh = dh;

        // pad to target_size rectangle left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        int top = static_cast<int>(std::round(dh - 0.1));
        int bottom = static_cast<int>(std::round(dh + 0.1));
        int left = static_cast<int>(std::round(dw - 0.1));
        int right = static_cast<int>(std::round(dw + 0.1));

        cv::Scalar color = cv::Scalar(114.f, 114.f, 114.f);
        cv::copyMakeBorder(input, input, top, bottom, left, right, cv::BORDER_CONSTANT, color);
        float* input_data = (float*)input.data;
        input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
        infer_request.set_input_tensor(input_tensor);
    }catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
}

void YOLOV9::postprocess_img(Mat& frame, float* detections, ov::Shape & output_shape) {
    std::vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    int out_rows = output_shape[1];
    int out_cols = output_shape[2];
    const cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)detections);
    for (int i = 0; i < det_output.cols; ++i) {
        const cv::Mat classes_scores = det_output.col(i).rowRange(4, 84);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > scoreThreshold) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow) );
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, this->scoreThreshold, this->nmsThreshold, nms_result);

    std::vector<Detection> output;
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result); 
    }
    cv::Mat image = frame.clone();
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        auto confidence = detection.confidence;

        // adjust offset to original unpadded
        float x0 = (box.x - imageshape.dw) / scale;
        float y0 = (box.y - imageshape.dh) / scale;
        float x1 = (box.x + box.width - imageshape.dw) / scale;
        float y1 = (box.y + box.height - imageshape.dh) / scale;

        //clip
        x0 = std::max(std::min(x0, (float)(imageshape.width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(imageshape.height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(imageshape.width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(imageshape.height - 1)), 0.f);

        box.x = x0;
        box.y = y0;
        box.width = x1 - x0;
        box.height = y1 - y0;

        // draw box
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        cv::Scalar color=  cv::Scalar(dis(gen),
            dis(gen),
            dis(gen));

        char text[256];
        sprintf(text, "%s %.1f%%", coconame[classId], confidence * 100);

        cv::rectangle(image, box, color, 3);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = box.x;
        int y = box.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imwrite("image.jpg", image);
}