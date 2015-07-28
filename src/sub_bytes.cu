#include "aes.h"
#include "aes.tab"

__global__ void aca_sub_bytes(void *state_buf)
{
  uint32_t *state = (uint32_t *)state_buf;
  size_t i = threadIdx.x;
  state[i] = sbox[state[i]];
}

__global__ void aca_inv_sub_bytes(void *state_buf)
{
  uint32_t *state = (uint32_t *)state_buf;
  size_t i = threadIdx.x;
  state[i] = inv_sbox[state[i]];
}
