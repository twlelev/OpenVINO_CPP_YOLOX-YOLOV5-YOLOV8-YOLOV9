# include"yolox.h"

int main() {

    Config config = { 0.2,0.4,0.4,640,640,"../optimized/yolox_s_int8.xml" };
    YOLOV5 yolomodel(config);
    clock_t start, end;
    cv::Mat img = cv::imread("../bus.jpg");
    start = clock();
    yolomodel.detect(img);
    end = clock();
    std::cout << "infer time = " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("result.jpg", img);
    cv::waitKey(0);

    // VideoCapture cap("2121.mp4");
    // if (!cap.isOpened()) return 1;
    // Mat frame;
    // while (true) {
    //     cap >> frame;
    //     start = clock();
    //     yolomodel.detect(frame);
    //     end = clock();
    //     cout << "infer time = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    //     imshow("image", frame);
    //     waitKey(1);
    // }
    return 0;
}

