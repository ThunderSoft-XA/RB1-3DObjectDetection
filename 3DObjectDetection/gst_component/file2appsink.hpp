#ifndef CAMERA_2_APPSINK_HPP
#define CAMERA_2_APPSINK_HPP

#include <iostream>
#include <fstream>
#include <string>

#include <gst/gst.h>
#include <gst/audio/audio.h>

#include <opencv2/opencv.hpp>
#include "easy_queue.hpp"

#include <json/json.h>

#define CHUNK_SIZE 1024   /* Amount of bytes we are sending in each buffer */
#define SAMPLE_RATE 44100 /* Samples per second we are sending */

/* Structure to contain all our information, so we can pass it to callbacks */
class CameraPipe {

private:
	GstElement *pipeline;
	GstElement *qtiqmmf, *qmmf_caps, *qtic2venc, *h264parse;
	GstElement *qtivdec, *videoconvert, *convert_caps, *app_sink;
	GstBus *bus;

	std::string pipe_name;
	int camera_id;
    int width, height, framerate;

	guint64 num_samples;   /* Number of samples generated so far (for timestamp generation) */
	gfloat a, b, c, d;     /* For waveform generation */

	guint sourceid;        /* To control the GSource */

public:
	CameraPipe(std::string json_file);
	int initPipe();

	int checkElements();

	void setProperty();
    // void setDataQueue(Queue<cv::Mat> &mat_queue) {
    //     rgb_mat_queue = mat_queue;
    // }

	int runPipeline();
    Queue<cv::Mat> rgb_mat_queue;

	~CameraPipe() {
		/* Free resources */
		gst_element_set_state (this->pipeline, GST_STATE_NULL);
		gst_object_unref (this->pipeline);
	}
};

/* This method is called by the idle GSource in the mainloop, to feed CHUNK_SIZE bytes into appsrc.
 * The idle handler is added to the mainloop when appsrc requests us to start sending data (need-data signal)
 * and is removed when appsrc has enough data (enough-data signal).
 */

using namespace Json;

/* The appsink has received a buffer */
static GstFlowReturn new_sample (GstElement *app_sink, gpointer data) 
{
    gint height;
    gint width;
    GstSample *sample;
    GstBuffer *sample_buffer;
    GstCaps *smaple_caps;
    const GstStructure *caps_struct;
    CameraPipe *pipe = static_cast<CameraPipe *>(data);

    /* Retrieve the buffer */
    g_signal_emit_by_name (app_sink, "pull-sample", &sample);
    if (sample){
        g_print("sample is valid, fill sample buffer to mat data\n");
        smaple_caps = gst_sample_get_caps (sample);
        if (!smaple_caps) {
            g_print ("gst_sample_get_caps fail\n");
            gst_sample_unref (sample);
            return GST_FLOW_ERROR;
        }
        caps_struct = gst_caps_get_structure (smaple_caps, 0);
        gboolean res;
        res = gst_structure_get_int (caps_struct, "width", &width);		//获取图片的宽
        res |= gst_structure_get_int (caps_struct, "height", &height);	//获取图片的高
        if (!res) {
            g_print ("gst_structure_get_int fail\n");
            gst_sample_unref (sample);
            return GST_FLOW_ERROR;
        }
        g_print ("sample caps width = %d, height = %d\n",width, height);

        //获取视频的一帧buffer,注意，这个buffer是无法直接用的，它不是char类型
        sample_buffer = gst_sample_get_buffer(sample);		

        if(!sample_buffer){
            g_print("gst_sample_get_buffer fail\n");
            gst_sample_unref (sample);
            return GST_FLOW_ERROR;
        }

        GstMapInfo map;
        //把buffer映射到map，这样我们就可以通过map.data取到buffer的数据
        if (gst_buffer_map (sample_buffer, &map, GST_MAP_READ)){		
            g_print("jpg size = %ld \n", map.size);
#if 0
            // nv12  convert to  RGB failed
            int yuvNV12_size = width * height * 3 / 2;
            cv::Mat yuvNV12,rgb24;
            yuvNV12.create(height * 3 / 2, width, CV_8UC1);
            memcpy(yuvNV12.data, map.data, yuvNV12_size);
            cv::cvtColor(yuvNV12, rgb24, cv::COLOR_YUV2RGB_NV12);
#endif

            cv::Mat sink_mat(cv::Size(width, height), CV_8UC3, map.data);
            g_print("this mat channels = %d\n", sink_mat.channels());
            pipe->rgb_mat_queue.push_back(sink_mat);
            cv::imwrite("rgb24.png",sink_mat);
            g_print("fill sample buffer to mat data is finished\n");
        }
        gst_buffer_unmap (sample_buffer, &map);	//解除映射
    }
    gst_sample_unref (sample);	//释放资源
    // GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(data->pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "capture");    //打印此时刻的管道状态图
    return GST_FLOW_OK;
}

/* This function is called when an error message is posted on the bus */
static void error_cb (GstBus *bus, GstMessage *msg, gpointer data) 
{
    GError *err;
    gchar *debug_info;

    /* Print error details on the screen */
    gst_message_parse_error (msg, &err, &debug_info);
    g_printerr("Error received from element %s: %s\n", GST_OBJECT_NAME (msg->src), err->message);
    g_printerr("Debugging information: %s\n", debug_info ? debug_info : "none");
    g_clear_error(&err);
    g_free(debug_info);

    // g_main_loop_quit (data->main_loop);
}

/** H264
gst-launch-1.0 -e qtiqmmfsrc camera=1 name=qmmf !\
    video/x-raw\(memory:GBM\),format=NV12,width=640,height=480,framerate=30/1 ! \
        qtic2venc ! appsink  sync=false  max-lateness=0 max-buffers=1 drop=true
**/

CameraPipe::CameraPipe(std::string json_file) 
{
    Json::Value root;
    std::ifstream ifs;
    ifs.open(json_file);

    CharReaderBuilder builder;
    builder["collectComments"] = true;
    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifs, &root, &errs)) {
        g_print("%s\n",std::string(errs).c_str());
    }
    g_print("parse json file %s successfully in %s\n",__FILE__,__FUNCTION__);

    int size = root.size();   // count of root node
    for (int index = 0; index < size; index++) {
    
        const Json::Value cameras_obj = root["gstreamer"]["cameras"];
        int cameras_size = cameras_obj.size();

        for(int cameras_index = 0; cameras_index < cameras_size; cameras_index++) {
            if (cameras_obj[cameras_index].isMember("pipe_name")) {
                this->pipe_name = cameras_obj[cameras_index]["pipe_name"].asString();
            }
            if (cameras_obj[cameras_index].isMember("camera_id")) {
                this->camera_id = cameras_obj[cameras_index]["camera_id"].asInt();
            }
            if (cameras_obj[cameras_index].isMember("width")) {
                this->width = cameras_obj[cameras_index]["width"].asInt();
            }
            if (cameras_obj[cameras_index].isMember("height")) {
                this->height = cameras_obj[cameras_index]["height"].asInt();
            }
            if (cameras_obj[cameras_index].isMember("framerate")) {
                this->framerate = cameras_obj[cameras_index]["framerate"].asInt();
            }
        }
    }
    g_print("camera info : pipe_name =  %s, camera_id = %d\n",this->pipe_name.c_str(), this->camera_id);
    g_print("\t image size: width = %d, height = %d\n",this->width, this->height);
    ifs.close();
    this->b = 1; /* For waveform generation */
    this->d = 1;

    g_print("end of %s constructor !!!\n", __FUNCTION__);

}

/*** NV12
 * gst-launch-1.0 -e qtiqmmfsrc camera=0 name=qmmf ! video/x-raw\(memory:GB
M\),format=NV12,width=640,height=480,framerate=30/1 ! qtic2venc ! h264parse  con
fig-interval=1 ! qtivdec ! qtivtransform ! video/x-raw,format=RGB ! appsink drop=true max-buffers=1 sync=false max-lateness=0
**/

/**
 * gst-launch-1.0 -e qtiqmmfsrc camera=0 name=qmmf ! video/x-raw\(memory:GBM\),
format=NV12,width=640,height=480,framerate=30/1 ! qtic2venc ! h264parse  config-
interval=1 ! qtivdec ! videoconvert ! video/x-raw,format=RGB ! appsink drop=true
 max-buffers=1 sync=false max-lateness=0
 * 
 */

int CameraPipe::initPipe()
{
    /* Create the elements */
    if( this->pipe_name.empty() ) {
        g_print("doing %s failed in %s\n !!!",__FUNCTION__,__FILE__);
        return -1;
    }

    this->qtiqmmf = gst_element_factory_make ("qtiqmmfsrc", "qmmf");
    this->qmmf_caps = gst_element_factory_make ("capsfilter", "qmmf_caps");
    this->qtic2venc = gst_element_factory_make ("qtic2venc", "c2venc");
    this->h264parse = gst_element_factory_make ("h264parse", "parser");
    this->qtivdec = gst_element_factory_make ("qtivdec", "qtivdec");
    this->videoconvert = gst_element_factory_make ("videoconvert", "convert");
    this->convert_caps = gst_element_factory_make ("capsfilter", "convert_caps");
    this->app_sink = gst_element_factory_make ("appsink", "app_sink");

    /* Create the empty camera pipeline */
    this->pipeline = gst_pipeline_new (this->pipe_name.c_str());
    g_print("doing %s  successfully in %s\n !!!",__FUNCTION__,__FILE__);

    return 0;
}


int CameraPipe::checkElements() 
{
    if (!this->pipeline) {
        g_print("element %s could be created.\n",gst_object_get_name(GST_OBJECT(this->pipeline)));
        return -1;
    }
    if (!this->qtiqmmf) {
        g_print("element %s could be created.\n",gst_object_get_name(GST_OBJECT(this->qtiqmmf)));
        return -1;
    }
    if (!this->qmmf_caps) {
        g_print("element %s could be created.\n",gst_object_get_name(GST_OBJECT(this->qmmf_caps)));
        return -1;
    }
    if (!this->qtic2venc) {
        g_print("element %s could be created.\n",gst_object_get_name(GST_OBJECT(this->qtic2venc)));
        return -1;
    }
    if (!this->h264parse) {
        g_print("element %s could be created.\n",gst_object_get_name(GST_OBJECT(this->h264parse)));
        return -1;
    }
    if (!this->qtivdec) {
        g_print("element %s could be created.\n",gst_object_get_name(GST_OBJECT(this->qtivdec)));
        return -1;
    }
    if (!this->videoconvert) {
        g_print("element %s could be created.\n",gst_object_get_name(GST_OBJECT(this->videoconvert)));
        return -1;
    }
    if (!this->convert_caps) {
        g_print("element %s could be created.\n",gst_object_get_name(GST_OBJECT(this->convert_caps)));
        return -1;
    }
    if (!this->app_sink) {
        g_print("element %s could be created.\n",gst_object_get_name(GST_OBJECT(this->app_sink)));
        return -1;
    }
    g_print("doing %s successfully in %s !!!\n", __FUNCTION__, __FILE__);
    return 0;
}

void CameraPipe::setProperty() 
{
    /* Configure wavescope */
    g_object_set(this->qtiqmmf, "camera", this->camera_id, NULL);
    g_print("doing %s successfully in %s - %d!!!\n", __FUNCTION__, __FILE__, __LINE__);

    // config capsfilter caps to qmmf
    //GstCaps *qmmf_caps_str = gst_caps_from_string("video/x-raw(memory:GBM), format=NV12, width=320, height=240, framerate=25/1");
    GstCaps *qmmf_caps_str = gst_caps_new_simple ("video/x-raw",
        "format", G_TYPE_STRING, "NV12",
        "width", G_TYPE_INT, this->width,
        "height", G_TYPE_INT, this->height,
        "framerate", GST_TYPE_FRACTION, this->framerate, 1,
        NULL);

    g_object_set(this->qmmf_caps, "caps", qmmf_caps_str, NULL);
    gst_caps_unref(qmmf_caps_str);
    g_print("doing %s successfully in %s - %d!!!\n", __FUNCTION__, __FILE__, __LINE__);

    // config h264parser
    g_object_set(this->h264parse, "config-interval", -1, NULL);
    g_print("doing %s successfully in %s - %d!!!\n", __FUNCTION__, __FILE__, __LINE__);

    // config capsfilter caps to transform
    GstCaps *convert_caps_str = gst_caps_from_string("video/x-raw,format=BGR");
    g_object_set(this->convert_caps, "caps", convert_caps_str, NULL);
    gst_caps_unref(convert_caps_str);

    g_print("doing %s successfully in %s - %d!!!\n", __FUNCTION__, __FILE__, __LINE__);

    /* Configure appsink */
    g_object_set(this->app_sink, "emit-signals", TRUE, NULL);
    g_object_set(this->app_sink, "sync",FALSE,NULL);
    g_object_set(this->app_sink, "max-lateness",0,NULL);
    g_object_set(this->app_sink, "max-buffers",1,NULL);
    g_object_set(this->app_sink, "drop",TRUE,NULL);

    g_print("doing %s successfully in %s !!!\n", __FUNCTION__, __FILE__);
}

int CameraPipe::runPipeline() {
    /* Link all elements that can be automatically linked because they have "Always" pads */
    gst_bin_add_many (GST_BIN (this->pipeline), this->qtiqmmf, this->qmmf_caps,
        this->qtic2venc, this->h264parse, this->qtivdec,
        this->videoconvert,  this->convert_caps, this->app_sink, NULL);
    if (gst_element_link_many ( this->qtiqmmf, this->qmmf_caps,
        this->qtic2venc, this->h264parse, this->qtivdec,
        this->videoconvert, this->convert_caps, this->app_sink, NULL) != TRUE) {
            g_printerr ("Elements could not be linked.\n");
            gst_object_unref (this->pipeline);
            return -1;
    }

    // callback
    g_signal_connect (this->app_sink, "new-sample", G_CALLBACK (new_sample), this);
    /* Instruct the bus to emit signals for each received message, and connect to the interesting signals */
    bus = gst_element_get_bus (this->pipeline);
    gst_bus_add_signal_watch (bus);
    g_signal_connect (G_OBJECT (bus), "message::error", (GCallback)error_cb, this);
    gst_object_unref (bus);

    g_print("doing %s successfully in %s !!!\n", __FUNCTION__, __FILE__);
    /* Start playing the pipeline */
    return gst_element_set_state (this->pipeline, GST_STATE_PLAYING);
        
}

#endif  // CAMERA_2_APPSINK_HPP