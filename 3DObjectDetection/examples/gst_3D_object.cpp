#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <thread>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include "inference_tf.hpp"
#include "camera2appsink.hpp"
#include "appsrc2rtsp.hpp"

using namespace inference;

extern Queue<cv::Mat> rgb_mat_queue;

int main(int argcm, char **argv)
{
    std::string json_file = "./gst_yolov5_config.json";

    /* Initialize GStreamer */
    gst_init (nullptr, nullptr);
    GMainLoop *main_loop;  /* GLib's Main Loop */
    /* Create a GLib Main Loop and set it to run */
    main_loop = g_main_loop_new (NULL, FALSE);
    Queue<cv::Mat> mat_queue;
    CameraPipe camera_pipe(json_file);

    camera_pipe.initPipe();
    camera_pipe.rgb_mat_queue = mat_queue;

    camera_pipe.checkElements();

    camera_pipe.setProperty();

    camera_pipe.runPipeline();
    Settings object_3d_setting;

    object_3d_setting.model_name = "./object_detection_3d_cup.tflite";
    object_3d_setting.input_bmp_name = "./cups.jpg";

    tf::TFInference object_3d_inference;

    object_3d_inference.setSettings(&object_3d_setting);

    object_3d_inference.loadModel();
    object_3d_inference.setInferenceParam();

    g_print("json_file : %s\n",json_file.c_str());
	Queue<cv::Mat> push_queue;
	APPSrc2RtspSink push_pipe(json_file);
	if( push_pipe.initPipe() == -1 ) {
		return 0;
	}

	push_pipe.push_mat_queue = push_queue;

	if(push_pipe.checkElements() == -1) {
		return 0;
	}

	push_pipe.setProperty();

    std::thread([&](){
        static int count = 0;
        while(1) {
            if(camera_pipe.rgb_mat_queue.empty()) {
                continue;
            }
            cv::Mat input_mat = camera_pipe.rgb_mat_queue.pop();
            std::stringstream input_name;
            input_name << "./input/" << count << ".png";
            cv::imwrite(input_name.str(),input_mat);

            tf::InputPair input_pair(0,input_mat);
            tf::InputPairVec input_pair_vec;//(object_3d_inference.getInputsNum());
            input_pair_vec.push_back(input_pair);
            object_3d_inference.loadData(input_pair_vec);
            std::vector<float> result_vec;
            object_3d_inference.inferenceModel<float>(result_vec);
            
            std::vector<cv::Point2f> cube_vec;
            for(int point_index = 0; point_index < result_vec.size() / 2; point_index++) {
                cv::Point2f point(result_vec[point_index * 2], result_vec[point_index * 2 + 1]);
                cube_vec.push_back(point);
            }
            cv::resize(input_mat, input_mat, cv::Size(224,224));

            cv::line(input_mat, cube_vec[1], cube_vec[3], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[1], cube_vec[2], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[1], cube_vec[5], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[4], cube_vec[3], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[4], cube_vec[2], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[4], cube_vec[8], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[6], cube_vec[2], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[6], cube_vec[5], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[6], cube_vec[8], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[7], cube_vec[3], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[7], cube_vec[5], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            cv::line(input_mat, cube_vec[7], cube_vec[8], 
                cv::Scalar(255, 255, 0), 2, cv::LINE_4);
            for(auto point : cube_vec) {
                cv::circle(input_mat, point, 3, cv::Scalar(0, 0, 255), -1);
            }

            cv::resize(input_mat,input_mat,{640,480});

            push_pipe.push_mat_queue.push_back(input_mat);
        }
    }).detach();

    sleep(4);
    std::thread([&](){
		push_pipe.runPipe();
    }).detach();

    g_main_loop_run(main_loop);

    camera_pipe.~CameraPipe();
    g_main_loop_unref (main_loop);

    return 0;

}