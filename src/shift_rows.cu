#include "aes.h"
__global__ void aca_shift_rows(aca_word_t *state)
{
  aca_size_t r = threadIdx.x;
  aca_size_t c, tmp[4];
  /* s[r,c] = state[r + 4c] */
  for(c = 0; c < 4; c++)
    tmp[c] = state[r + 4 * ((c+r) % Nb)];
  for(c = 0; c < 4; c++)
    state[r + 4*c] = tmp[c];
}

__global__ void aca_inv_shift_rows(aca_word_t * state)
{
  aca_size_t r = threadIdx.x;
  aca_size_t c,tmp[4];

  for(c = 0; c < 4; c++)
    tmp[c] = state[r + 4*c];
  for(c = 0; c < 4; c++)
    state[r + 4 * ((c+r) % Nb)] = tmp[c];

}
