MediaServer &

ffmpeg4 -re -i udp://192.168.0.1:8900 -vcodec copy -f rtsp -rtsp_transport tcp rtsp://192.168.0.1:554/live/test &
ffmpeg4 -re -i udp://192.168.0.1:8901 -vcodec copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:554/live/test1 &

gst-launch-1.0 -e qtiqmmfsrc camera=0 name=qmmf ! video/x-raw\(memory:GBM\),format=NV12,width=640,height=480,framerate=30/1 ! qtic2venc ! h264parse  config-interval=1 ! mpegtsmux name=muxer ! queue ! udpsink port=8900 host=192.168.0.1 &
gst-launch-1.0 -e qtiqmmfsrc camera=1 name=qmmf ! video/x-raw\(memory:GBM\),format=NV12,width=640,height=480,framerate=30/1 ! qtic2venc ! h264parse  config-interval=1 ! mpegtsmux name=muxer ! queue ! udpsink port=8901 host=192.168.0.1 &