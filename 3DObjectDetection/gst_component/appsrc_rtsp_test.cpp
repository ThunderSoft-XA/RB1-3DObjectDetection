#include <unistd.h>
#include <thread>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <opencv2/opencv.hpp>

#include "appsrc2rtsp.hpp"
#include "utils.hpp"

gint main (gint argc, gchar **argv)
{
	std::string json_file = "./gst_yolov5_config.json";
    // if( parse_arg(argc, argv, json_file) != 0) {
	// 	g_print("invalid argument, exit !!\n");
	// 	return -1;
	// }
	gst_init (nullptr, nullptr);
    GMainLoop *loop;
	loop = g_main_loop_new (NULL, FALSE);

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

	cv::Mat depth_estimation_mat = cv::imread("./left.jpg");
	g_print("mat data size = %d\n", (int)depth_estimation_mat.total());
    cv::resize(depth_estimation_mat,depth_estimation_mat,{640,480});
	push_pipe.updateCaps(depth_estimation_mat.cols,depth_estimation_mat.rows);

	std::thread([&](){
        while( 1 ) {
			push_pipe.push_mat_queue.push_back(depth_estimation_mat);
			sleep(0.03);
			// usleep(40000);
        }
    }).detach();
	sleep(4);

	g_print(" push_queue size = %d\n" ,push_pipe.push_mat_queue.size());
	std::thread([&](){
		push_pipe.runPipe();
    }).detach();
	g_main_loop_run(loop);

	push_pipe.~APPSrc2RtspSink();
	g_main_loop_unref (loop);

	return 0;
}


