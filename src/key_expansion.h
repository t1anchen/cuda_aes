#include "commons.h"

/*
 * Interfaces of Cryptographics
 */
#ifndef ACA_CUDA_AES_KEY_EXPANSION_H
#define ACA_CUDA_AES_KEY_EXPANSION_H
extern "C" void aca_key_expansion(void *key_buf, size_t key_len, void *W_buf, size_t Nk, size_t Nr);
void aca_inv_key_expansion(void *key, size_t key_len, void *W, size_t Nk, size_t Nr);
#endif /* ACA_CUDA_AES_KEY_EXPANSION_H */
