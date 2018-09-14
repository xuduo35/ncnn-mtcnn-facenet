#include "mtcnn.h"
#include "mobilefacenet.h"
#include <opencv2/opencv.hpp>

using namespace cv;

#define MAXFACEOPEN 0 //设置是否开关最大人脸调试，1为开，其它为关

void test_picture(char *samplejpg){
    const char *model_path = "../models";
    MTCNN mtcnn(model_path);

    clock_t start_time = clock();

    cv::Mat image;
    image = cv::imread(samplejpg);
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
    std::vector<Bbox> finalBbox;

#if(MAXFACEOPEN==1)
    mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
    mtcnn.detect(ncnn_img, finalBbox);
#endif

    const int num_box = finalBbox.size();
    std::vector<cv::Rect> bbox;
    bbox.resize(num_box);

    if (num_box == 1) {
        cv::Mat ROI(image, Rect(finalBbox[0].x1, finalBbox[0].y1, finalBbox[0].x2 - finalBbox[0].x1 + 1, finalBbox[0].y2 - finalBbox[0].y1 + 1));

        cv::Mat croppedImage;

        // Copy the data into new matrix
        ROI.copyTo(croppedImage);

        imwrite("sample.jpg",croppedImage);
    } else {
        std::cout << "no face detected or too much faces" << std::endl;
    }
#if 0
    for (int i = 0; i < num_box; i++) {
        bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

        for (int j = 0; j<5; j = j + 1)
        {
            cv::circle(image, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
        }
    }

    for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
        rectangle(image, (*it), Scalar(0, 0, 255), 2, 8, 0);
    }

    imshow("face_detection", image);
    clock_t finish_time = clock();
    double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "time" << total_time * 1000 << "ms" << std::endl;

    cv::waitKey(0);
#endif
}

void test_video(int argc, char** argv) {
    const char *model_path = "../models";
    MTCNN mtcnn(model_path);
    mtcnn.SetMinFace(40);
    cv::VideoCapture mVideoCapture(argc>1?argv[1]:0);
    if (!mVideoCapture.isOpened()) {
        return;
    }
    cv::Mat frame;
    mVideoCapture >> frame;
    while (!frame.empty()) {
        mVideoCapture >> frame;
        if (frame.empty()) {
            break;
        }

        clock_t start_time = clock();

        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
        std::vector<Bbox> finalBbox;
#if(MAXFACEOPEN==1)
        mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
        mtcnn.detect(ncnn_img, finalBbox);
#endif
        const int num_box = finalBbox.size();
        std::vector<cv::Rect> bbox;
        bbox.resize(num_box);
        for(int i = 0; i < num_box; i++){
            bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

            for (int j = 0; j<5; j = j + 1)
            {
                cv::circle(frame, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
            }
        }
        for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
            rectangle(frame, (*it), Scalar(0, 0, 255), 2, 8, 0);
        }
        imshow("face_detection", frame);
        clock_t finish_time = clock();
        double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
        std::cout << "time" << total_time * 1000 << "ms" << std::endl;

        int q = cv::waitKey(10);
        if (q == 27) {
            break;
        }
    }
    return ;
}

void test_facenet(int argc, char** argv) {
    const char *model_path = "../models";
    Recognize recognize(model_path);

    cv::Mat sampleimg = cv::imread("./sample.jpg", CV_LOAD_IMAGE_COLOR);

    std::vector<float> samplefea;

    clock_t start_time = clock();
    recognize.start(sampleimg, samplefea);

    clock_t finish_time = clock();
    double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "time" << total_time * 1000 << "ms" << std::endl;

    MTCNN mtcnn(model_path);
    mtcnn.SetMinFace(40);
    cv::VideoCapture mVideoCapture(argc>2?argv[2]:0);
    if (!mVideoCapture.isOpened()) {
        return;
    }
    cv::Mat frame;
    mVideoCapture >> frame;
    while (!frame.empty()) {
        mVideoCapture >> frame;
        if (frame.empty()) {
            break;
        }

        clock_t start_time = clock();

        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
        std::vector<Bbox> finalBbox;
#if(MAXFACEOPEN==1)
        mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
        mtcnn.detect(ncnn_img, finalBbox);
#endif
        const int num_box = finalBbox.size();
        std::vector<cv::Rect> bbox;
        bbox.resize(num_box);

        // for(int i = 0; i < num_box; i++){
        //     bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

        //     for (int j = 0; j<5; j = j + 1)
        //     {
        //         cv::circle(frame, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
        //     }
        // }

        for(int i = 0; i < num_box; i++){
            cv::Rect r = Rect(finalBbox[0].x1, finalBbox[0].y1, finalBbox[0].x2 - finalBbox[0].x1 + 1, finalBbox[0].y2 - finalBbox[0].y1 + 1);
            cv::Mat ROI(frame, r);

            cv::Mat croppedImage;
            std::vector<float> croppedfea;

            // Copy the data into new matrix
            ROI.copyTo(croppedImage);

            recognize.start(croppedImage, croppedfea);

            double similar = calculSimilar(samplefea, croppedfea);

            // std::cout << "similarity is : " << similar << std::endl;

            if (similar > 0.65) {
                rectangle(frame, r, Scalar(0, 0, 255), 2, 8, 0);
            } else {
                for (int j = 0; j<5; j = j + 1)
                {
                    cv::circle(frame, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
                }
            }

        }

        imshow("face_detection", frame);
        clock_t finish_time = clock();
        double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
        std::cout << "time" << total_time * 1000 << "ms" << std::endl;

        if (cv::waitKey(10) == 27) {
            break;
        }
    }
    return ;
}

int main(int argc, char** argv) {
    if ((argc > 1) && !strcmp(argv[1], "-sample")) {
        test_picture(argv[2]);
        return 0;
    } else if ((argc > 1) && !strcmp(argv[1], "-facenet")) {
        test_facenet(argc, argv);
        return 0;
    }

    test_video(argc, argv);
    //test_picture();
    return 0;
}
/*
int main()
{
    char *model_path = "../models";
    Recognize recognize(model_path);

    cv::Mat img1 = cv::imread("../pic/Aaron_Tippin_0001.jpg", CV_LOAD_IMAGE_COLOR);
    cv::Mat img2 = cv::imread("../pic/Aaron_Peirsol_0004.jpg", CV_LOAD_IMAGE_COLOR);
    std::vector<float> feature1;
    std::vector<float> feature2;

    clock_t start_time = clock();
    recognize.start(img1, feature1);
    recognize.start(img2, feature2);
    double similar = calculSimilar(feature1, feature2);
    clock_t finish_time = clock();
    double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;

    std::cout << "time" << total_time * 1000 << "ms" << std::endl;
    std::cout << "similarity is : " << similar << std::endl;
    cv::imshow("left", img1);
    cv::imshow("right", img2);
    cv::waitKey(0);

    return 0;
}
*/