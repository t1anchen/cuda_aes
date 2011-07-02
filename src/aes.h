#ifndef AU9CUDA_AES_H
#define AU9CUDA_AES_H


/* 
 * Necessary Includes 
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


/* 
 * Specification of Data Presentation
 */
#ifndef ACA_TYPE
#define ACA_TYPE
typedef unsigned char aca_byte_t;
typedef unsigned int aca_word_t;
typedef unsigned int aca_size_t;
typedef signed int aca_ssize_t;
typedef unsigned int u32;
typedef unsigned char u8;
#endif	

#ifdef ACA_BLOCK_BITS_SIZE
#undef ACA_BLOCK_BITS_SIZE
#endif
#define ACA_BLOCK_BITS_SIZE 128

#ifdef ACA_BLOCK_BYTES_SIZE
#undef ACA_BLOCK_BYTES_SIZE
#endif
#define ACA_BLOCK_BYTES_SIZE 16

#ifdef ACA_MIN_KEY_BITS_SIZE
#undef ACA_MIN_KEY_BITS_SIZE
#endif
#define ACA_MIN_KEY_BITS_SIZE 128

#ifdef ACA_MIN_KEY_BYTES_SIZE
#undef ACA_MIN_KEY_BYTES_SIZE
#endif
#define ACA_MIN_KEY_BYTES_SIZE 16

#ifdef ACA_MAX_KEY_BITS_SIZE
#undef ACA_MAX_KEY_BITS_SIZE
#endif
#define ACA_MAX_KEY_BITS_SIZE 256
	
#ifdef ACA_MAX_KEY_BYTES_SIZE
#undef ACA_MAX_KEY_BYTES_SIZE
#endif
#define ACA_MAX_KEY_BYTES_SIZE 32

#ifdef ACA_MAX_KEY_WORDS
#undef ACA_MAX_KEY_WORDS
#endif
#define ACA_MAX_KEY_WORDS (ACA_MAX_KEY_BITS_SIZE/32)

#ifdef ACA_MAX_KEY_BYTES
#undef ACA_MAX_KEY_BYTES
#endif
#define ACA_MAX_KEY_BYTES (ACA_MAX_KEY_BITS_SIZE/8)

#ifdef ACA_MAX_NR
#undef ACA_MAX_NR
#endif
#define ACA_MAX_NR 14

const aca_size_t size = 4 * 4 * sizeof(aca_word_t);
const aca_size_t Nb = 4;


/*
 * Define include files when benchmark 
 */
#ifdef ACA_BENCHMARK
#include <time.h>
#endif


/* 
 * Inline Funtions (Macro Implementation)
 */
#define xtime_byte(x) ((x<<1) ^ (((x>>7) & 1) * 0x1b))
#define xtime_word(x) ((((x) & 0x7f7f7f7fU) << 1) ^ ((((x) & 0x80808080U) >> 7) * 0x1b))
#define ror32(w,shift)				\
  (((w) >> (shift)) | ((w) << (32 - (shift))))
#define GET(M,X,Y) ((M)[((Y) << 2) + (X)])


/*
 * Errno Specification 
 */
#ifndef ACA_ERRNO
#define ACA_ERRNO
enum aca_errno_t {
  NORMAL = 0,
  ABNORMAL = 1
};
enum aca_op_t {
  NONE       = 0,
  ENCRYPT    = 1,
  DECRYPT    = 2,
  CBC_MODE   = 4,
  ECB_MODE   = 8,
  ASYNC_MODE = 16
};
#endif


/* 
 * Interfaces of Cryptographics
 */
void aca_key_expansion(aca_word_t *key, aca_size_t key_len, aca_word_t *W, aca_size_t Nk, aca_size_t Nr);
void aca_inv_key_expansion(aca_word_t *key, aca_size_t key_len, aca_word_t *W, aca_size_t Nk, aca_size_t Nr);
__global__ void aca_sub_bytes(aca_word_t *state);
__global__ void aca_inv_sub_bytes(aca_word_t *state);
__global__ void aca_shift_rows(aca_word_t *state);
__global__ void aca_inv_shift_rows(aca_word_t * state);
__global__ void aca_mix_columns(aca_word_t *state);
__global__ void aca_inv_mix_columns(aca_word_t *state);
__global__ void aca_add_round_key(aca_word_t *state, aca_word_t *key);
void aca_aes_encrypt_core(aca_word_t *cp, aca_word_t *cW, aca_word_t Nr);
void aca_aes_decrypt_core(aca_word_t *cp, aca_word_t *cW, aca_word_t Nr);
void aca_aes_encrypt(aca_word_t *pt, aca_word_t *key, aca_word_t *ct, aca_word_t keysize);
void aca_aes_decrypt(aca_word_t *pt, aca_word_t *key, aca_word_t *ct, aca_word_t keysize);


/* 
 * Interfaces of Utilities
 */
void my_cp_print_hexbytes(aca_word_t *bytes, aca_size_t bytes_len);
aca_size_t my_str2bytes(aca_word_t **dst, const char *src);
void my_print_hexbytes(aca_word_t *bytes, aca_size_t bytes_len);


#endif	/* AU9CUDA_AES_H */

