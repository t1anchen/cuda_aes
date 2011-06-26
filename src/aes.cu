#include "aes.h"
#include "aes.tab"


static const aca_word_t Rcon[] = { 0, 0x01000000, 0x02000000, 0x04000000, 0x08000000,
				   0x10000000, 0x20000000, 0x40000000, 0x80000000,
				   0x1B000000, 0x36000000 };

#define ror32(w,shift)				\
  (((w) >> (shift)) | ((w) << (32 - (shift))))

#define aca_shift(r,nb) ((r)%(nb))

#define GET(M,X,Y) ((M)[((Y) << 2) + (X)])

const aca_size_t size = 4 * 4 * sizeof(aca_word_t);
/* static inline aca_word_t ror32(aca_word_t w, aca_size_t shift) */
/* { */
/*   return (w >> shift) | (w << (32 - shift)); */
/* } */

/* Implementations of the prototypes in the header file */
const aca_size_t Nb = 4;

void my_cp_print_hexbytes(aca_word_t *bytes, aca_size_t bytes_len)
{
  aca_size_t i;
  for(i = 0; i < bytes_len; i++)
    printf("%X", bytes[i]);
  printf("\n");
}


__global__ void aca_sub_bytes(aca_word_t *state)
{
  aca_size_t i = threadIdx.x;
  state[i] = sbox[state[i]];
}

__global__ void aca_inv_sub_bytes(aca_word_t *state)
{
  aca_size_t i = threadIdx.x;
  state[i] = inv_sbox[state[i]];
}

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
  aca_word_t t, Tmp, Tm;
  aca_word_t u, v, w;

  u = state[base];
  v = xtime_byte(u);
  w = xtime_byte(v);
  t = w ^ state[base];
  t = t ^ state[base];


  /* aca_word_t buf=0,t,u,v,w,y; */


  /* buf = (state[base] & 0xff) | ((state[base + 1] & 0xff) << 8) | ((state[base + 2] & 0xff) << 16) | ((state[base + 3]) << 24); */

  /* u = xtime_word(buf); */
  /* v = xtime_word(u); */
  /* w = xtime_word(v); */
  /* t = w ^ buf; */
  /* y = u ^ v ^ w; */
  /* y ^= ror32(u ^ t, 8) ^ ror32(v ^ t, 16) ^ ror32(t, 24); */

  /* state[base] = y & 0xffU; */
  /* state[base + 1] = (y & 0xff00U) >> 8; */
  /* state[base + 2] = (y & 0xff0000U) >> 16; */
  /* state[base + 3] = (y & 0xff000000U) >> 24; */
}

__global__ void aca_add_round_key(aca_word_t *state, aca_word_t *key)
{
  aca_size_t i = threadIdx.x;
  state[i] ^= key[i];
}

/* __global__ void aca_sub_word(aca_word_t *state) */
/* { */

/* } */

void aca_key_expansion(aca_word_t *key, aca_size_t key_len, aca_word_t *W, aca_size_t Nk, aca_size_t Nr)
{
  uint i, j, cols, temp, tmp[4];
  cols = (Nr + 1) << 2;

  memcpy(W, key, (key_len >> 3)*sizeof(uint));

  for(i=Nk; i<cols; i++) {
    for(j=0; j<4; j++)
      tmp[j] = GET(W, j, i-1);
    if(Nk > 6) {
      if(i % Nk == 0) {
	temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
	tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
	tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
	tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
	tmp[3] = temp;
      } else if(i % Nk == 4) {
	tmp[0] = hsbox[tmp[0]];
	tmp[1] = hsbox[tmp[1]];
	tmp[2] = hsbox[tmp[2]];
	tmp[3] = hsbox[tmp[3]];
      }
    } else {
      if(i % Nk == 0) {
	temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
	tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
	tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
	tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
	tmp[3] = temp;
      }
    }
    for(j=0; j<4; j++)
      GET(W, j, i) = GET(W, j, i-Nk) ^ tmp[j];
  }

}

void aca_inv_key_expansion(aca_word_t *key, aca_size_t key_len, aca_word_t *W, aca_size_t Nk, aca_size_t Nr)
{
  uint i, j, cols, temp, tmp[4];
  cols = (Nr + 1) << 2;

  memcpy(W, key, (key_len >> 3)*sizeof(uint));

  for(i=Nk; i<cols; i++) {
    for(j=0; j<4; j++)
      tmp[j] = GET(W, j, i-1);
    if(Nk > 6) {
      if(i % Nk == 0) {
	temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
	tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
	tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
	tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
	tmp[3] = temp;
      } else if(i % Nk == 4) {
	tmp[0] = hsbox[tmp[0]];
	tmp[1] = hsbox[tmp[1]];
	tmp[2] = hsbox[tmp[2]];
	tmp[3] = hsbox[tmp[3]];
      }
    } else {
      if(i % Nk == 0) {
	temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
	tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
	tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
	tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
	tmp[3] = temp;
      }
    }
    for(j=0; j<4; j++)
      GET(W, j, i) = GET(W, j, i-Nk) ^ tmp[j];
  }

  for(i = 1; i < Nr; i++)
    aca_inv_mix_columns<<<1,4>>>(W+(i<<4));


}

void aca_aes_encrypt_core(aca_word_t *cp, aca_word_t *cW, aca_word_t Nr)
{
  aca_word_t i;
  aca_add_round_key<<<1,16>>>(cp, cW);
  for(i = 1; i < Nr; i++) {
    aca_sub_bytes<<<1,16>>>(cp);
    aca_shift_rows<<<1,4>>>(cp);
    aca_mix_columns<<<1,4>>>(cp);
    aca_add_round_key<<<1,16>>>(cp, cW+(i << 4));
  }
  aca_sub_bytes<<<1,16>>>(cp);
  aca_shift_rows<<<1,4>>>(cp);
  aca_add_round_key<<<1,16>>>(cp, cW+(i << 4));
}

void aca_aes_decrypt_core(aca_word_t *cp, aca_word_t *cW, aca_word_t Nr)
{
  aca_word_t i;
  aca_add_round_key<<<1,16>>>(cp, cW+(Nr<<4));
  for(i = Nr-1; i >=1; i--) {
    aca_inv_sub_bytes<<<1,16>>>(cp);
    aca_inv_shift_rows<<<1,4>>>(cp);
    aca_inv_mix_columns<<<1,4>>>(cp);
    aca_add_round_key<<<1,16>>>(cp, cW+(i << 4));
  }
  aca_inv_sub_bytes<<<1,16>>>(cp);
  aca_inv_shift_rows<<<1,4>>>(cp);
  aca_add_round_key<<<1,16>>>(cp, cW);
}

void aca_aes_encrypt(aca_word_t *pt, aca_word_t *key, aca_word_t *ct, aca_word_t keysize)
{
  aca_word_t *cp, *W, *cW, Nk, Nr;
  Nk = keysize >> 5;
  Nr = Nk + 6;

  aca_word_t s = ((Nr+1) * sizeof(aca_word_t)) << 4;
  W = (aca_word_t *)malloc(s);
  cudaMalloc((void**)&cW, s);
  aca_key_expansion(key, keysize, W, Nk, Nr);
  cudaMemcpy(cW, W, s, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&cp, size);
  cudaMemcpy(cp, pt, size, cudaMemcpyHostToDevice);

  aca_aes_encrypt_core(cp, cW, Nr);

  cudaMemcpy(ct, cp, size, cudaMemcpyDeviceToHost);
}

void aca_aes_decrypt(aca_word_t *pt, aca_word_t *key, aca_word_t *ct, aca_word_t keysize)
{
  aca_word_t *cp, *W, *cW, Nk, Nr;
  Nk = keysize >> 5;
  Nr = Nk + 6;

  aca_word_t s = ((Nr+1) * sizeof(aca_word_t)) << 4;
  W = (aca_word_t *)malloc(s);
  cudaMalloc((void**)&cW, s);
  aca_inv_key_expansion(key, keysize, W, Nk, Nr);
  cudaMemcpy(cW, W, s, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&cp, size);
  cudaMemcpy(cp, pt, size, cudaMemcpyHostToDevice);

  aca_aes_decrypt_core(cp, cW, Nr);

  cudaMemcpy(ct, cp, size, cudaMemcpyDeviceToHost);
}
