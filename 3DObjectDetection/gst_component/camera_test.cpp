#include "camera2appsink.hpp"

#include "utils.hpp"

int main(int argc, char *argv[])
{
	std::string json_file = "./gst_yolov5_config.json";
    // if( parse_arg(argc, argv, json_file) != 0) {
	// 	std::cout << "invalid argument, exit !!" << std::endl;
	// 	return -1;
	// }
     /* Initialize GStreamer */
    gst_init (nullptr, nullptr);
    GMainLoop *main_loop;  /* GLib's Main Loop */
    /* Create a GLib Main Loop and set it to run */
    main_loop = g_main_loop_new (NULL, FALSE);
    
    CameraPipe camera_pipe(json_file);

    camera_pipe.initPipe();

    camera_pipe.checkElements();

    camera_pipe.setProperty();

    camera_pipe.runPipeline();

    g_main_loop_run (main_loop);

    camera_pipe.~CameraPipe();
    g_main_loop_unref (main_loop);
    return 0;
}