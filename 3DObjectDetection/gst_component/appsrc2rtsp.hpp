#ifndef APPSRC_2_RTSP_HPP
#define APPSRC_2_RTSP_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <list>
#include <stdlib.h>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <opencv2/opencv.hpp>

#include "easy_queue.hpp"
#include "json/json.h"

static int caps_width;
static int caps_height;
static int caps_framerate;

static cv::Mat default_mat;

class APPSrc2RtspSink {

private: 
	GstElement *pipeline, *appsrc, *videoconvert, *convert_caps, *qtic2venc, *h264parse, *rtph264pay,*mpegtsmux, *rtspsink;
	GstAppSrcCallbacks cbs;
	/* Pointer to hold bus address */
	GstBus *bus = nullptr;

	std::string pipe_name;
	int width, height, framerate;
	std::string host = "127.0.0.1";
	int port = 5000;
public:
	GMainLoop *loop;
	guint sourceid;        /* To control the GSource */

	APPSrc2RtspSink(std::string json_file);
	int initPipe();
	Queue<cv::Mat> push_mat_queue;

	int checkElements();

	void setProperty();
	void updateCaps(int width, int height);


	int runPipe();
	~APPSrc2RtspSink();

};


static void cb_need_data(GstAppSrc *appsrc, guint unused_size, gpointer user_data)
{
	g_print("In %s\n", __func__);
	static GstClockTime timestamp = 0;
    APPSrc2RtspSink *pipe = static_cast<APPSrc2RtspSink *>(user_data);
	GstBuffer *buffer;
	guint size;
	GstFlowReturn ret;
	GstMapInfo map;

	size = caps_width * caps_height * 3;
    g_print("check fact caps size : %d\n",size );

	buffer = gst_buffer_new_allocate (NULL, size, NULL);
    static gboolean white = FALSE;

	/* this makes the image black/white */
	// gst_buffer_memset (buffer, 0, white ? 0xff : 0x0, size);
	// gst_buffer_memset (buffer, 0, *data, size);

//这两个方法都可以
#if 0
	g_print("In %s: %i\n", __FILE__,__LINE__);
    gst_buffer_fill(buffer, 0, &user_data, size);
#else
    gst_buffer_map (buffer, &map, GST_MAP_WRITE);
	g_print("In %s: %i\n", __FILE__,__LINE__);
	g_print("map.data address = %p\n",map.data);
	g_print("user_data address = %p\n",user_data);
	g_print("size = %d\n", size);//gst_buffer_get_size(buffer)
    cv::Mat mat_data;
    if(pipe->push_mat_queue.empty()) {
        g_print("push_mat_queue is empty,so using default mat\n");
        /* this makes the image black/white */
        if(default_mat.empty()) {
            gst_buffer_memset (buffer, 0, white ? 0xFF : 0x00, size);
        } else {
            mat_data = default_mat;
            memcpy( (guchar *)map.data, (guchar *)mat_data.data, size);
        }
    } else {
        mat_data = pipe->push_mat_queue.pop();
        if(pipe->push_mat_queue.empty()) {
            pipe->push_mat_queue.push_back(mat_data);
        }
        if (mat_data.empty()) {
            g_print("mat from push_mat_queue is empty ,so using default mat\n");
            if(default_mat.empty()) {
                gst_buffer_memset(buffer, 0, white ? 0xFF : 0x00, size);
            } else {
                mat_data = default_mat;
                memcpy( (guchar *)map.data, (guchar *)mat_data.data, size);
            }
        } else {
            mat_data.copyTo(default_mat);
            memcpy( (guchar *)map.data, (guchar *)mat_data.data, size);
        }
    }

#endif
    //set time stamp
	GST_BUFFER_PTS (buffer) = timestamp;
	GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int (1, GST_SECOND, caps_framerate);
	timestamp += GST_BUFFER_DURATION (buffer);

	g_signal_emit_by_name (appsrc, "push-buffer", buffer, &ret);
    //ret = gst_app_src_push_buffer(appsrc, buffer);

    gst_buffer_unmap (buffer, &map);	//解除映射

	gst_buffer_unref (buffer);	//释放资源

	if (ret != GST_FLOW_OK) {
		/* something wrong, stop pushing */
		// g_main_loop_quit (pipe->loop);
	}
    g_print("free end, over \n");
}

static void cb_enough_data(GstAppSrc *src, gpointer user_data)
{
	g_print("In %s\n", __func__);
}

static gboolean cb_seek_data(GstAppSrc *src, guint64 offset, gpointer user_data)
{
	g_print("In %s\n", __func__);
	return TRUE;
}

static gboolean cb_bus (GstBus *bus, GstMessage *message, gpointer data) 
{
	g_print("Handle gstreamer pipeline bus message\n");
	GstElement *pipeline = (GstElement *)(data);
	switch (GST_MESSAGE_TYPE (message)) {
	case GST_MESSAGE_ERROR: {
		GError *err;
		gchar *debug;

		gst_message_parse_error (message, &err, &debug);
		g_print ("Error: %s\n", err->message);
		g_error_free (err);
		// g_free (debug);

		gst_element_set_state(pipeline,GST_STATE_READY);
		break;
	} case GST_MESSAGE_EOS:
		/* end-of-stream */
		gst_element_set_state(pipeline,GST_STATE_NULL);
		break;
	default:
	/* unhandled message */
		break;
	}
	/* we want to be notified again the next time there is a message
	* on the bus, so returning TRUE (FALSE means we want to stop watching
	* for messages on the bus and our callback should not be called again)
	*/
	return TRUE;
}

/* This signal callback triggers when appsrc needs data. Here, we add an idle handler
 * to the mainloop to start pushing data into the appsrc */
static void start_feed (GstElement *source, guint size, APPSrc2RtspSink *data) {
  if (data->sourceid == 0) {
    g_print ("Start feeding\n");
    data->sourceid = g_idle_add ((GSourceFunc) cb_need_data, data);
  }
}

/* This callback triggers when appsrc has enough data and we can stop sending.
 * We remove the idle handler from the mainloop */
static void stop_feed (GstElement *source, APPSrc2RtspSink *data) {
  if (data->sourceid != 0) {
    g_print ("Stop feeding\n");
    g_source_remove (data->sourceid);
    data->sourceid = 0;
  }
}

APPSrc2RtspSink::APPSrc2RtspSink(std::string json_file) 
{
    Json::Value root;
    std::ifstream ifs;
    ifs.open(json_file);

    Json::CharReaderBuilder builder;
    builder["collectComments"] = true;
    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifs, &root, &errs)) {
        g_print("%s\n", std::string(errs).c_str());
    }
    g_print("parse json file %s, successfully in %s\n",json_file.c_str(),__FUNCTION__ );
    ifs.close();
    
    int size = root.size();   // count of root node
    for (int index = 0; index < size; index++) {
    
        const Json::Value streams_obj = root["gstreamer"]["streams"];
        int streams_size = streams_obj.size();

        for(int streams_index = 0; streams_index < streams_size; streams_index++) {
            if (streams_obj[streams_index].isMember("pipe_name")) {
                this->pipe_name = streams_obj[streams_index]["pipe_name"].asString();
            }
            if (streams_obj[streams_index].isMember("width")) {
                caps_width = this->width = streams_obj[streams_index]["width"].asInt();
            }
            if (streams_obj[streams_index].isMember("height")) {
                caps_height = this->height = streams_obj[streams_index]["height"].asInt();
            }
            if (streams_obj[streams_index].isMember("framerate")) {
                caps_framerate = this->framerate = streams_obj[streams_index]["framerate"].asInt();
            }
            if (streams_obj[streams_index].isMember("host")) {
                this->host = streams_obj[streams_index]["host"].asString();
            }
            if (streams_obj[streams_index].isMember("port")) {
                this->port = streams_obj[streams_index]["port"].asInt();
            }
        }
    }
    this->sourceid = 0;

    g_print("json configure info : %s = %d x %d \n",this->pipe_name.c_str(),this->width,this->height);
    g_print("IP: Host = %s, port = %d\n",this->host.c_str(),this->port);
    g_print("end of %s constructor !!!\n", __FUNCTION__);
}

/**
 * @brief "appsrc ! videoconvert ! x264enc ! rtspclientsink location=rtsp://localhost:8554/live/test_red"
 * 
 */

int APPSrc2RtspSink::initPipe()
{
    if( this->pipe_name.empty() ) {
        g_print("doing %s failed in %s\n",__FILE__,__FUNCTION__);
        return -1;
    }
    pipeline = gst_pipeline_new (this->pipe_name.c_str());
    appsrc = gst_element_factory_make ("appsrc", "source");
	videoconvert = gst_element_factory_make("videoconvert", "convert");
    convert_caps = gst_element_factory_make("capsfilter", "filter");
	qtic2venc = gst_element_factory_make("x264enc", "c2venc"); //qtic2venc x264enc
	rtspsink = gst_element_factory_make("rtspclientsink", "sink");

    g_print("doing %s successfully in %s\n",__FILE__,__FUNCTION__);
    return 0;
}


int APPSrc2RtspSink::checkElements() 
{
    if (!this->pipeline) {
        g_print("element pipeline could be created." );
        return -1;
    }
    if (!this->appsrc) {
        g_print("element appsrc could be created." );
        return -1;
    }
    if (!this->videoconvert) {
        g_print("element videoconvert could be created." );
        return -1;
    }
    if (!this->convert_caps) {
        g_print("element convert_caps could be created." );
        return -1;
    }   
    if (!this->qtic2venc) {
        g_print("element qtic2venc could be created." );
        return -1;
    }
    if (!this->rtspsink) {
        g_print("element rtspsink could be created." );
        return -1;
    }

    g_print("doing %s successfully in %s\n",__FILE__,__FUNCTION__);
    return 0;
}

void APPSrc2RtspSink::setProperty() 
{
    /* setup */
    g_object_set (G_OBJECT (appsrc), "caps",
        gst_caps_new_simple ("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, this->width,
            "height", G_TYPE_INT, this->height,
            "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1,
            "framerate", GST_TYPE_FRACTION, this->framerate, 1,
            NULL),
        NULL);
    g_object_set (G_OBJECT (appsrc),
        "stream-type", 0,
        "is-live", TRUE,
        "do-timestamp", TRUE,
        "min-latency", 0,
        "format", GST_FORMAT_TIME, NULL);

    GstCaps *convert_caps_str  = gst_caps_new_simple ("video/x-raw",
        "format", G_TYPE_STRING, "NV12",NULL);
    g_object_set(G_OBJECT(this->convert_caps), "caps", convert_caps_str, NULL);
    gst_caps_unref(convert_caps_str);

    std::stringstream location;
    location << "rtsp://" << this->host << ":"
        << std::to_string(this->port) << "/live/test";
    g_print("location: %s\n",location.str().c_str());
    g_object_set(G_OBJECT(this->rtspsink), "location",  
        "rtsp://localhost:/live/test", NULL);
	// g_object_set(this->rtspsink, "sync", false, NULL);
	// g_object_set(this->rtspsink, "async", false, NULL);

    this->cbs.need_data = cb_need_data;
    this->cbs.enough_data = cb_enough_data;
    this->cbs.seek_data = cb_seek_data;

    g_print("doing %s successfully in %s\n",__FILE__,__FUNCTION__);
}

void APPSrc2RtspSink::updateCaps(int width, int height)
{
    if( (width != this->width) || (height != this->height)) {
        /* update caps */
        g_print("caps size different, updating ......" );
        caps_width = this->width = width;
        caps_height = this->height = height;
        g_object_set (G_OBJECT (appsrc), "caps",
        gst_caps_new_simple ("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, this->width,
            "height", G_TYPE_INT, this->height,
            "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1,
            "framerate", GST_TYPE_FRACTION, caps_framerate, 1,
            NULL), NULL);
    }

    g_print("doing %s successfully in %s\n",__FILE__,__FUNCTION__);
}

int APPSrc2RtspSink::runPipe()
{
    /* TEST WITH rtspsink */
    /* NOT WORKING AS EXPECTED, DISPLAYING A BLURRY VIDEO */
    gst_bin_add_many (GST_BIN (this->pipeline), this->appsrc, this->videoconvert, this->convert_caps,
        this->qtic2venc, this->rtspsink, NULL);

    if (gst_element_link_many ( this->appsrc, this->videoconvert, this->convert_caps,
        this->qtic2venc, this->rtspsink, NULL)!= TRUE) {
            g_printerr ("Elements could not be linked.\n");
            gst_object_unref (pipeline);
            return -1;
    }
    // gst_app_src_set_callbacks(GST_APP_SRC_CAST(appsrc), &cbs, depth_estimation_mat.data, NULL);
    g_signal_connect (this->appsrc, "need-data", G_CALLBACK (cb_need_data), this);
    g_signal_connect (this->appsrc, "enough-data", G_CALLBACK (stop_feed), this);

    bus = gst_element_get_bus(pipeline);
    gst_bus_add_signal_watch (bus);
    g_signal_connect (G_OBJECT (bus), "message::error", (GCallback)cb_bus, this);
    gst_object_unref(bus);

    /* play */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("doing %s successfully in %s\n",__FILE__,__FUNCTION__);
    return 0;
}

APPSrc2RtspSink::~APPSrc2RtspSink()
{
    /* clean up */
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
}

#endif // APPSRC_2_RTSP_HPP