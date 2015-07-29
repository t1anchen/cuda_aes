#include "aes.h"

__global__ void aca_add_round_key(void *state_buf, void *key_buf)
{
  uint32_t *state = (uint32_t *)state_buf;
  uint32_t *key = (uint32_t *)key_buf;
  size_t i = threadIdx.x;
  state[i] ^= key[i];
}
