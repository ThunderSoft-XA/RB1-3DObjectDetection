#!/bin/bash

# set android device
adb wait-for-usb-device && adb root && adb remount && adb shell mount -o remount,rw /

PROJECT_DIR=`pwd`

echo ${PROJECT_DIR}

adb -s 1e9da838 push ${PROJECT_DIR}/../env/hostapd.conf /etc/wlan/
adb -s 1e9da838 push ${PROJECT_DIR}/../env/udhcpd.conf /etc/

adb -s 1e9da838 push ${PROJECT_DIR}/../bin/gst_test /data
adb -s 1e9da838 push ${PROJECT_DIR}/../bin/gst_test_server /data

adb -s 1e9da838 shell ifconfig wlan0 192.168.0.1 netmask 255.255.255.0
adb -s 1e9da838 shell hostapd -B /etc/wlan/hostapd.conf

adb -s 1e9da838 shell mkdir -p /var/lib/misc
adb -s 1e9da838 shell touch /var/lib/misc/udhcpd.leases
adb -s 1e9da838 shell udhcpd -f /etc/udhcpd.conf  &


