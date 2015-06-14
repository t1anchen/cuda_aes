/* File: cuda_aes_for_py.i */

%module cuda_aes_for_py

%{
#define SWIG_FILE_WITH_INIT
#include "cuda_aes_for_py.h"
#include "utils.h"
%}
int kick_off(int argc, char*argv[]);
void my_cp_print_hexbytes(aca_word_t *bytes, aca_size_t bytes_len);
aca_size_t my_str2bytes(aca_word_t **dst, const char *src);
void my_print_hexbytes(aca_word_t *bytes, aca_size_t bytes_len);


