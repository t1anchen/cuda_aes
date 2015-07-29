#include "cipher.h"
#include "key_expansion.h"
#include "aes.h"

/* Export these functions as public interface */
extern "C" {
  void aca_aes_encrypt(void *pt, void *key, void *ct, size_t keysize);
  void aca_aes_decrypt(void *pt, void *key, void *ct, size_t keysize);
  void aca_aes_print_helloworld();
}

static size_t size = 4 * 4 * sizeof(uint32_t);

static void 
aca_aes_encrypt_core(void *input_ptr, void *input_words, size_t Nr)
{
  size_t i;
  uint32_t *cp = (uint32_t *)input_ptr;
  uint32_t *cW = (uint32_t *)input_words;
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
aca_aes_decrypt_core(void *input_ptr, void *input_words, size_t Nr)
{
  size_t i;
  uint32_t *cp = (uint32_t *)input_ptr;
  uint32_t *cW = (uint32_t *)input_words;
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
aca_aes_encrypt(void *pt, void *key, void *ct, size_t keysize)
{
  uint32_t *cp, *W, *cW;
  size_t Nk, Nr;
  Nk = keysize >> 5;
  Nr = Nk + 6;

  size_t block_len = ((Nr+1) * sizeof(uint32_t)) << 4; /* 4*(Nr+1) words */

  /* Allocate memory for key both host buffer and device buffer */
  W = (uint32_t *)malloc(block_len);
  cudaMalloc((void**)&cW, block_len);

  /* Key expansion (on host)*/
  aca_key_expansion(key, keysize, W, Nk, Nr);

  /* Move key to device */
  cudaMemcpy(cW, W, block_len, cudaMemcpyHostToDevice);

  /* Allocate momery for encrypted message buffer (only 128) */
  cudaMalloc((void**)&cp, size);
  cudaMemcpy(cp, pt, size, cudaMemcpyHostToDevice);

  /* Encrypt */
  aca_aes_encrypt_core(cp, cW, Nr);

  /* Move cipher to host (only 128) */
  cudaMemcpy(ct, cp, size, cudaMemcpyDeviceToHost);
}

void 
aca_aes_decrypt(void *pt, void *key, void *ct, size_t keysize)
{
  uint32_t *cp, *W, *cW;
  size_t Nk, Nr;
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
