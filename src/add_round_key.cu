#include "aes.h"

__global__ void aca_add_round_key(aca_word_t *state, aca_word_t *key)
{
  aca_size_t i = threadIdx.x;
  state[i] ^= key[i];
}
