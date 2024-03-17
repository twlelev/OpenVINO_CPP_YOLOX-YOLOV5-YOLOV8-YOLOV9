# include"yolov9.h"

int main(int argc, char* argv[]) {
    
    try{
        if(argc!=3){
            std::cout<<"Usage:"<<argv[0]<<" <path_to_model> <path_to_image>"<<std::endl;
            return EXIT_FAILURE;
        }
        const std::string input_model_path {argv[1]};
        const std::string input_image_path {argv[2]}; 
        Config config = { 0.2,0.4,0.45,640,640, input_model_path}; 
        clock_t start, end;
        cv::Mat img = cv::imread(input_image_path);

        YOLOV9 yolomodel(config);
        start = clock();
        yolomodel.detect(img);
        end = clock();
        std::cout << "infer time = " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
    }catch (const std::exception& ex){
        std::cerr << ex.what()<<std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}