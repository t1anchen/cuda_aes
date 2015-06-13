/* File: cuda_aes_for_py.i */

%module cuda_aes_for_py

%{
#define SWIG_FILE_WITH_INIT
#include "cuda_aes_for_py.h"
#include "utils.h"
#include "cipher.h"
%}
int kick_off(int argc, char*argv[]);


