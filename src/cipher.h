#include "commons.h"

#ifndef ACA_CUDA_AES_CIPHER_H
#define ACA_CUDA_AES_CIPHER_H
void aca_aes_encrypt(aca_word_t *pt, aca_word_t *key, aca_word_t *ct, aca_word_t keysize);
void aca_aes_decrypt(aca_word_t *pt, aca_word_t *key, aca_word_t *ct, aca_word_t keysize);
#endif /* ACA_CUDA_AES_CIPHER_H */
