#include "aes.h"

__global__ void aca_add_round_key(uint32_t *state, uint32_t *key)
{
  aca_size_t i = threadIdx.x;
  state[i] ^= key[i];
}
