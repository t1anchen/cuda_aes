#include "commons.h"

/* 
 * Interfaces of Cryptographics
 */
#ifndef ACA_CUDA_AES_KEY_EXPANSION_H
#define ACA_CUDA_AES_KEY_EXPANSION_H
void aca_key_expansion(aca_word_t *key, aca_size_t key_len, aca_word_t *W, aca_size_t Nk, aca_size_t Nr);
void aca_inv_key_expansion(aca_word_t *key, aca_size_t key_len, aca_word_t *W, aca_size_t Nk, aca_size_t Nr);
#endif /* ACA_CUDA_AES_KEY_EXPANSION_H */
