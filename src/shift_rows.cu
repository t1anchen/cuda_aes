#include "aes.h"

__global__ void aca_shift_rows(void *state_buf)
{
  uint32_t *state = (uint32_t *)state_buf;
  size_t r = threadIdx.x;
  size_t c, tmp[4];
  /* s[r,c] = state[r + 4c] */
  for(c = 0; c < 4; c++)
    tmp[c] = state[r + 4 * ((c+r) % 4)];
  for(c = 0; c < 4; c++)
    state[r + 4*c] = tmp[c];
}

__global__ void aca_inv_shift_rows(void * state_buf)
{
  uint32_t *state = (uint32_t *)state_buf;
  size_t r = threadIdx.x;
  size_t c,tmp[4];

  for(c = 0; c < 4; c++)
    tmp[c] = state[r + 4*c];
  for(c = 0; c < 4; c++)
    state[r + 4 * ((c+r) % 4)] = tmp[c];

}
