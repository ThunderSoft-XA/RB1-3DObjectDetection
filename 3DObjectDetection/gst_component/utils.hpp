#ifndef UTILS_HPP
#define UTILS_HPP

#include <stdlib.h>
#include <getopt.h>

#include <iostream>
#include <string>

static struct option longopts[] = {
	{"help", no_argument, NULL, 'h'},
	{"json_file", required_argument, NULL, 'f'},
	{0, 0, 0, 0},
};

int parse_arg(int argc, char **argv, std::string &json_file)
{
	int ret = -1;
	char c = '0';
    char g_file[256] = {0};
    printf("parsing json file ......\n");
	while ( (c = getopt_long(argc, argv, "-:f:h", longopts, NULL)) != -1) {
		switch (c) {
            case 'f':
                snprintf(g_file, sizeof(g_file), "%s", optarg);
                json_file = std::string(g_file);
                ret = 0;
                break;
            case '?':
                fprintf(stdout, "%c invalid argument\n", optopt);
                break;
            default:
                break;
		}
    }
    return ret;
}

#endif //UTILS_HPP