#include "commons.h"

/* 
 * Interfaces of Cryptographics
 */
__global__ void aca_sub_bytes(aca_word_t *state);
__global__ void aca_inv_sub_bytes(aca_word_t *state);
__global__ void aca_shift_rows(aca_word_t *state);
__global__ void aca_inv_shift_rows(aca_word_t * state);
__global__ void aca_mix_columns(aca_word_t *state);
__global__ void aca_inv_mix_columns(aca_word_t *state);
__global__ void aca_add_round_key(aca_word_t *state, aca_word_t *key);

