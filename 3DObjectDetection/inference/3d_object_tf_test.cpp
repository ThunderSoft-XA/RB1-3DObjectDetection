#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "inference_tf.hpp"

using namespace inference;

int main(int argcm, char **argv)
{
    Settings object_3d_setting;

    object_3d_setting.model_name = "./object_detection_3d_cup.tflite";
    object_3d_setting.input_bmp_name = "./cups.jpg";

    std::cout << "\nobject_3d_setting.input_bmp_name : " << object_3d_setting.input_bmp_name << std::endl;

    tf::TFInference object_3d_inference;

    object_3d_inference.setSettings(&object_3d_setting);

    object_3d_inference.loadModel();
    object_3d_inference.setInferenceParam();

    cv::Mat object_3d_mat = cv::imread(object_3d_setting.input_bmp_name);

    cv::imwrite("object_3d_mat.png",object_3d_mat);
    tf::InputPair input_pair(0,object_3d_mat);
    tf::InputPairVec input_pair_vec;//(object_3d_inference.getInputsNum());
    input_pair_vec.push_back(input_pair);
    object_3d_inference.loadData(input_pair_vec);
    std::vector<float> result_vec;
    object_3d_inference.inferenceModel<float>(result_vec);
    std::cout << "result_vec size : " << result_vec.size() << std::endl;

    // 224, 224, 3
    // result_mat.data = (uchar *)object_3d_result_vec.data();
    // result_mat.convertTo(result_mat,CV_8UC3);
    // cv::cvtColor(result_mat, result_mat, cv::COLOR_RGB2BGR);

    std::vector<cv::Point2f> cube_vec;
    for(int point_index = 0; point_index < result_vec.size() / 2; point_index++) {
        cv::Point2f point(result_vec[point_index * 2], result_vec[point_index * 2 + 1]);
        cube_vec.push_back(point);
    }
    cv::resize(object_3d_mat, object_3d_mat, cv::Size(224,224));

    cv::line(object_3d_mat, cube_vec[1], cube_vec[3], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[1], cube_vec[2], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[1], cube_vec[5], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[4], cube_vec[3], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[4], cube_vec[2], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[4], cube_vec[8], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[6], cube_vec[2], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[6], cube_vec[5], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[6], cube_vec[8], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[7], cube_vec[3], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[7], cube_vec[5], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    cv::line(object_3d_mat, cube_vec[7], cube_vec[8], 
        cv::Scalar(255, 255, 0), 2, cv::LINE_4);
    for(auto point : cube_vec) {
        cv::circle(object_3d_mat, point, 3, cv::Scalar(0, 0, 255), -1);
    }
    cv::imwrite("./result_mat.png",object_3d_mat);

    return 0;

}