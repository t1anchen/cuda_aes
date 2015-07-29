#include "commons.h"

/* 
 * Interfaces of Cryptographics
 */
#ifndef ACA_CUDA_AES_ACTIONS_H
#define ACA_CUDA_AES_ACTIONS_H
__global__ void aca_sub_bytes(void *state);
__global__ void aca_inv_sub_bytes(void *state);
__global__ void aca_shift_rows(void *state);
__global__ void aca_inv_shift_rows(void * state);
__global__ void aca_mix_columns(void *state);
__global__ void aca_inv_mix_columns(void *state);
__global__ void aca_add_round_key(void *state, void *key);
#endif /* ACA_CUDA_AES_ACTIONS_H */
