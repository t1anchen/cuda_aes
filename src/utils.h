#include "commons.h"

/* 
 * Interfaces of Utilities
 */
#ifndef ACA_CUDA_AES_UTILS_H
#define ACA_CUDA_AES_UTILS_H
void my_cp_print_hexbytes(aca_word_t *bytes, aca_size_t bytes_len);
aca_size_t my_str2bytes(aca_word_t **dst, const char *src);
void my_print_hexbytes(uint32_t *bytes, uint32_t bytes_len);
#endif /* ACA_CUDA_AES_CIPHER_H */
