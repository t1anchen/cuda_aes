#include "aes.h"
__global__ void aca_mix_columns(aca_word_t *state)
{
  aca_word_t col  = threadIdx.x;
  aca_word_t base = col << 2;
  aca_word_t t, Tmp, Tm;

  t = state[base];
  Tmp = state[base] ^ state[base + 1] ^ state[base + 2] ^ state[base + 3];
  Tm = state[base] ^ state[base + 1]; Tm = xtime_byte(Tm) & 0xff; state[base] ^= Tm ^ Tmp;
  Tm = state[base + 1] ^ state[base + 2]; Tm = xtime_byte(Tm) & 0xff; state[base + 1] ^= Tm ^ Tmp;
  Tm = state[base + 2] ^ state[base + 3]; Tm = xtime_byte(Tm) & 0xff; state[base + 2] ^= Tm ^ Tmp;
  Tm = state[base + 3] ^ t;      Tm = xtime_byte(Tm) & 0xff; state[base + 3] ^= Tm ^ Tmp;
}

__global__ void aca_inv_mix_columns(aca_word_t *state)
{
  aca_word_t col = threadIdx.x;
  aca_word_t base = col << 2;
  aca_word_t t, Tmp;
  aca_word_t u, v, w;

  Tmp = state[base] ^ state[base + 1] ^ state[base + 2] ^ state[base + 3];
  u = xtime_byte(Tmp) & 0xff;
  v = xtime_byte(u) & 0xff;
  w = xtime_byte(v) & 0xff;

  t = w ^ Tmp;
  t ^= (xtime_byte((xtime_byte(state[base]) & 0xff)) & 0xff) ^ state[base];
  t ^= (xtime_byte(state[base + 1]) & 0xff);
  t ^= (xtime_byte((xtime_byte(state[base + 2]) & 0xff)) & 0xff);
  state[base] = t;

  t = w ^ Tmp;
  t ^= (xtime_byte((xtime_byte(state[base+1]) & 0xff)) & 0xff) ^ state[base+1];
  t ^= (xtime_byte(state[base+2]) & 0xff);
  t ^= (xtime_byte((xtime_byte(state[base+3]) & 0xff)) & 0xff);
  state[base+1] = t;

  t = w ^ Tmp;
  t ^= (xtime_byte((xtime_byte(state[base+2]) & 0xff)) & 0xff) ^ state[base+2];
  t ^= (xtime_byte(state[base + 3]) & 0xff);
  t ^= (xtime_byte((xtime_byte(state[base]) & 0xff)) & 0xff);
  state[base+2] = t;

  t = w ^ Tmp;
  t ^= (xtime_byte((xtime_byte(state[base+3]) & 0xff)) & 0xff) ^ state[base+3];
  t ^= (xtime_byte(state[base]) & 0xff);
  t ^= (xtime_byte((xtime_byte(state[base+1]) & 0xff)) & 0xff);
  state[base+3] = t;
}
