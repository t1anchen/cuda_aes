#include "cipher.h"
#include "key_expansion.h"
#include "aes.h"

/* Export these 2 functions as public interface */
extern "C" void aca_aes_encrypt(aca_word_t *pt, aca_word_t *key, aca_word_t *ct, aca_word_t keysize);
extern "C" void aca_aes_decrypt(aca_word_t *pt, aca_word_t *key, aca_word_t *ct, aca_word_t keysize);

static aca_size_t size = 4 * 4 * sizeof(aca_word_t);

static void aca_aes_encrypt_core(aca_word_t *cp, aca_word_t *cW, aca_word_t Nr)
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

static void aca_aes_decrypt_core(aca_word_t *cp, aca_word_t *cW, aca_word_t Nr)
{
  aca_word_t i;
  aca_add_round_key<<<1,16>>>(cp, cW);
  for(i = Nr-1; i >=1; i--) {
    aca_inv_sub_bytes<<<1,16>>>(cp);
    aca_inv_shift_rows<<<1,4>>>(cp);
    aca_inv_mix_columns<<<1,4>>>(cp);
    aca_add_round_key<<<1,16>>>(cp, cW+(i << 4));
  }
  aca_inv_sub_bytes<<<1,16>>>(cp);
  aca_inv_shift_rows<<<1,4>>>(cp);
  aca_add_round_key<<<1,16>>>(cp, cW+(i << 4));
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
