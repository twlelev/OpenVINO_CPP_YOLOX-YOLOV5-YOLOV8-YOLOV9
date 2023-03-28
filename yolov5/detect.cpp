# include"yolov5.h"
# include<string>

// openvino、yolov5、cpu、onnx；
int main(int argc, char const *argv[]) {
    try{
        if(argc!=2){
            std::cout<< "Usage:"<<argv[0]<<"path_to_image"<<std::endl;
            return EXIT_FAILURE;
        }
        const std::string input_image_path {argv[1]};

        // confThreshold;nmsThreshold;scoreThreshold;inpWidth;inpHeight;onnx_path;
        Config config = {0.4,0.4,0.4,640,640,"../optimized/yolov5n_int8.xml"};  
        YOLOV5 yolomodel(config);
        clock_t start, end;
        cv::Mat img=cv::imread(input_image_path);
        yolomodel.detect(img);
        imwrite("result.jpg", img);
    }catch (const std::exception& ex){
        std::cerr << ex.what()<<std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
