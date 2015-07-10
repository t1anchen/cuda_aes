#include "cipher.h"
#include "key_expansion.h"
#include "aes.h"

/* Export these functions as public interface */
extern "C" {
  void aca_aes_encrypt(uint32_t *pt, uint32_t *key, uint32_t *ct, uint32_t keysize);
  void aca_aes_decrypt(uint32_t *pt, uint32_t *key, uint32_t *ct, uint32_t keysize);
  void aca_aes_print_helloworld();
}

static aca_size_t size = 4 * 4 * sizeof(uint32_t);

static void 
aca_aes_encrypt_core(uint32_t *cp, uint32_t *cW, uint32_t Nr)
{
  uint32_t i;
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

static void 
aca_aes_decrypt_core(uint32_t *cp, uint32_t *cW, uint32_t Nr)
{
  uint32_t i;
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

void 
aca_aes_encrypt(uint32_t *pt, uint32_t *key, uint32_t *ct, uint32_t keysize)
{
  uint32_t *cp, *W, *cW, Nk, Nr;
  Nk = keysize >> 5;
  Nr = Nk + 6;

  uint32_t s = ((Nr+1) * sizeof(uint32_t)) << 4;
  W = (uint32_t *)malloc(s);
  cudaMalloc((void**)&cW, s);
  aca_key_expansion(key, keysize, W, Nk, Nr);
  cudaMemcpy(cW, W, s, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&cp, size);
  cudaMemcpy(cp, pt, size, cudaMemcpyHostToDevice);

  aca_aes_encrypt_core(cp, cW, Nr);

  cudaMemcpy(ct, cp, size, cudaMemcpyDeviceToHost);
}

void 
aca_aes_decrypt(uint32_t *pt, uint32_t *key, uint32_t *ct, uint32_t keysize)
{
  uint32_t *cp, *W, *cW, Nk, Nr;
  Nk = keysize >> 5;
  Nr = Nk + 6;

  uint32_t s = ((Nr+1) * sizeof(uint32_t)) << 4;
  W = (uint32_t *)malloc(s);
  cudaMalloc((void**)&cW, s);
  aca_inv_key_expansion(key, keysize, W, Nk, Nr);
  cudaMemcpy(cW, W, s, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&cp, size);
  cudaMemcpy(cp, pt, size, cudaMemcpyHostToDevice);

  aca_aes_decrypt_core(cp, cW, Nr);

  cudaMemcpy(ct, cp, size, cudaMemcpyDeviceToHost);
}

void
aca_aes_print_helloworld()
{
  void *an_int_buffer_on_gpu = NULL;
  cudaMalloc((void **)&an_int_buffer_on_gpu, 0x10);
  printf("a buffer with 0x10 length has been allocated by CUDA\n");
}
