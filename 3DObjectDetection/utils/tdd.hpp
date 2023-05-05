#include <iostream>
#include <cstdio>
#include <cstdlib>

void error(char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(EXIT_FAILURE);
}

void error_code(char *msg, int code)
{
    fprintf(stderr, "ERROR: %s (0x%x)\n", msg, code);
    exit(EXIT_FAILURE);
}

void error_code_fail(const char *file, int line, const char *expr, int code)
{
    fprintf(stderr, "%s(%d): %s failed with 0x%x\n", file, line, expr, code);
    exit(EXIT_FAILURE);
}

#define TEST_ERROR_CODE(x) { int e; if ( (e = x) != 0 ) { error_code_fail(__FILE__, __LINE__, #x, e); } }