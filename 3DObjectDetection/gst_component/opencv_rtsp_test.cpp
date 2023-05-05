#include <unistd.h>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <opencv2/opencv.hpp>

int main(int argc,char **argv) 
{
    cv::Mat frame;

    cv::String gst_str = "appsrc ! videoconvert ! x264enc ! rtspclientsink location=rtsp://localhost:/live/test protocols=tcp sync=false";
    // cv::String gst_str = "appsrc ! videoconvert ! x264enc ! filesink location=x264_test.h264";
    //cv::String gst_str = "appsrc ! videoconvert ! omxh264enc ! rtspclientsink location=rtsp://localhost:8554/live/test_red";
    //PC上没有omxh264enc插件，嵌入式平台有
    // 软编码 x264enc  ，硬编码 qtic2venc

    cv::VideoWriter video_writer(gst_str,cv::CAP_GSTREAMER, 0, 30, cv::Size(640,480));

    int id = 0;

    // 将 cv 读到的每一帧传入子进程
    frame = cv::imread("./left.jpg");
    cv::resize(frame,frame,{640,480});
    while (1)
    {
        // cv::String strFrameId;

        // id = (++id) % 10000;

        // QString idStr = QString("%1").arg(id);

        // strFrameId = idStr.toStdString();

        // cv::putText(frame, strFrameId, cv::Point(960,500), cv::HersheyFonts::FONT_ITALIC, 10, cv::Scalar(0,255,0), 3);
        if(frame.empty()) {
            continue;
        }

        std::cout << frame.size() << std::endl;
        sleep(0.03);

        video_writer.write(frame);
    }

	return 0;

}