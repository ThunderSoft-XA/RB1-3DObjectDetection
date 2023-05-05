//
// Created by liqing on 18-8-31.
//

#ifndef _TIMEUTIL_H
#define _TIMEUTIL_H

#include <sys/time.h>

#include <time.h>

#define GET_FRAME_RATE(start_time,frame_count,frame_rate) \
    struct timeval end_time; \
    double elapsed_time;    \
    gettimeofday(&end_time, nullptr); \
    elapsed_time = (end_time.tv_sec - start_time->tv_sec) * 1000 +  \
                                    (end_time.tv_usec - start_time->tv_usec) / 1000;    \
    if(elapsed_time > 1000*10) {    \
        frame_rate = frame_count * 1000 / elapsed_time; \
        memcpy(start_time,&end_time,sizeof(struct timeval)); \
        frame_count = 0;    \
    }


inline long getCurrentTime_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

inline long getCurrentTime_s() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec;
}

#include <iostream>
#include <sstream>
template <typename T>
inline std::string rdc_int2string(T n) {
    std::stringstream ss;
    std::string str;
    ss << n;
    ss >> str;
    return str;
}

inline std::string getCurrentTimeStr() {
    // get current system time
    time_t now;
    time(&now);
    struct tm* local_tm = localtime(&now);
    std::string year = rdc_int2string(local_tm->tm_year + 1900);
    std::string month = rdc_int2string(local_tm->tm_mon + 1);
    std::string day = rdc_int2string(local_tm->tm_mday);
    std::string hour = rdc_int2string(local_tm->tm_hour);
    std::string minute = rdc_int2string(local_tm->tm_min);
    std::string second = rdc_int2string(local_tm->tm_sec);
    std::string str;
    str.append(year).append("-").append(month).append("-").append(day)
            .append("-").append(hour).append("-").append(minute).append("-")
            .append(second);
    return str;
}

#include <stdlib.h>
#include <ctime>
inline int getRandNum(int num) {
    srand(time(0));
    return (rand() % num);
}

inline std::string GetLocalTimeWithMs(void)
{
    std::string defaultTime = "1970-01-01 00:00:00:000";
    try
    {
        struct timeval curTime;
        gettimeofday(&curTime, NULL);
        int milli = curTime.tv_usec / 1000;

        char buffer[80] = {0};
        struct tm nowTime;
        localtime_r(&curTime.tv_sec, &nowTime);//For thread security 
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &nowTime);

        char currentTime[84] = {0};
        snprintf(currentTime, sizeof(currentTime), ":%s:%03d", buffer, milli);

        return currentTime;
    }
    catch(const std::exception& e)
    {
        return defaultTime;
    }
    catch (...)
    {
        return defaultTime;
    }
}

inline long getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

inline double get_us(struct timeval t) 
{ 
    return (t.tv_sec * 1000000 + t.tv_usec); 
}

#endif //_TIMEUTIL_H
