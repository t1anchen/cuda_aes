#ifndef AU9CUDA_AES_H
#define AU9CUDA_AES_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Define customized types */
#ifndef ACA_TYPE
#define ACA_TYPE
typedef unsigned char aca_byte_t;
typedef unsigned int aca_word_t;
typedef unsigned int aca_size_t;
typedef signed int aca_ssize_t;
#endif	

/* Define data blocksize */
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


/* Define include files when benchmark */
#ifdef ACA_BENCHMARK
#include <time.h>
#endif

/* Specify the macro operation 
 * NOTE: The endianness of these operation is big-endian. */
#ifndef ACA_GETWORD
#define ACA_GETWORD(pt)				\
  (((aca_word_t)(pt)[0] << 24) ^			\
   ((aca_word_t)(pt)[1] << 16) ^			\
   ((aca_word_t)(pt)[2] << 8) ^			\
   ((aca_word_t)(pt)[3]))
#endif
#ifndef ACA_SETWORD
#define ACA_SETWORD(ct,st)			\
  ((ct)[0] = (aca_byte_t)((st) >> 24),		\
   (ct)[1] = (aca_byte_t)((st) >> 16),		\
   (ct)[2] = (aca_byte_t)((st) >> 8),		\
   (ct)[3] = (aca_byte_t)(st),			\
   (st))
#endif
#ifndef ACA_SWAP
#define ACA_SWAP(x,y)				\
  ((x)^=(y),					\
   (y)^=(x),					\
   (x)^=(y))
#endif
#ifndef ACA_ROL
#define ACA_ROL(x,y)					\
  ((x) = (((x) << (y)) | ((x) >> (32-(y)))) & 0xffffffffUL)
#endif
#ifndef ACA_ROLWORD
#define ACA_ROLWORD(x) \
  ((x) = (((x) << 8) | ((x) >> 24) & 0xffffffffUL))
#endif

#define xtime_byte(x) ((x<<1) ^ (((x>>7) & 1) * 0x1b))
#define xtime_word(x) ((((x) & 0x7f7f7f7fU) << 1) ^ ((((x) & 0x80808080U) >> 7) * 0x1b))

/* Errno Specification */
#ifndef ACA_ERRNO
#define ACA_ERRNO
enum aca_errno_t {
  NORMAL = 0,
  ABNORMAL = 1
};

/* au9cuda_aes options */
enum aca_op_t {
  NONE       = 0,
  ENCRYPT    = 1,
  DECRYPT    = 2,
  CBC_MODE   = 4,
  ECB_MODE   = 8,
  ASYNC_MODE = 16
};
#endif

/* interfaces */
void aca_key_expansion(aca_word_t *key, aca_size_t key_len, aca_word_t* W, aca_size_t Nk, aca_size_t Nr);
__global__ void aca_sub_bytes(aca_word_t *state);
__global__ void aca_inv_sub_bytes(aca_word_t *state);
__global__ void aca_shift_rows(aca_word_t *state);
__global__ void aca_inv_shift_rows(aca_word_t * state);
__global__ void aca_mix_colomns(aca_word_t *state);
__global__ void aca_inv_mix_colomns(aca_word_t *state);
__global__ void aca_add_round_key(aca_word_t *state, aca_word_t *key);
void aca_aes_encrypt_core(aca_word_t *cp, aca_word_t *cW, aca_word_t Nr);
void aca_aes_decrypt_core(aca_word_t *cp, aca_word_t *cW, aca_word_t Nr);
void aca_aes_encrypt(aca_word_t *pt, aca_word_t *key, aca_word_t *ct, aca_word_t keysize);
void aca_aes_decrypt(aca_word_t *pt, aca_word_t *key, aca_word_t *ct, aca_word_t keysize);


#endif	/* AU9CUDA_AES_H */

