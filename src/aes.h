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

#define xtime(x) ((x<<1) ^ (((x>>7) & 1) * 0x1b))

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

/* Method Specification */
/* aca_errno_t aca_scheduled_mem_init(aca_byte_t *sched); */
/* aca_errno_t aca_scheduled_mem_destory(aca_byte_t *sched); */
/* aca_errno_t aca_u32s_to_bytes(aca_byte_t *dst, const aca_word_t *src); */
/* aca_errno_t aca_bytes_to_u32s(aca_word_t *dst, const aca_byte_t *src); */
/* aca_errno_t aca_make_key(aca_word_t *dst, const aca_byte_t *src, aca_word_t key_bitsize, aca_op_t ops); */
aca_errno_t aca_encrypt(aca_word_t *dst, const aca_word_t *src, aca_op_t ops);
aca_errno_t aca_decrypt(aca_word_t *dst, const aca_word_t *src, aca_op_t ops);

/* Operation Details */
aca_errno_t aca_key_expansion(aca_word_t *w, aca_word_t *key, aca_size_t key_len, aca_size_t Nk, aca_size_t Nr);
aca_errno_t aca_inverse_cipher(aca_byte_t *dst, const aca_byte_t *src, const aca_word_t *W);
__global__ void aca_sub_bytes(aca_word_t *state);
__global__ void aca_shift_rows(aca_word_t *state);
__global__ void aca_mix_colomns(aca_word_t *state);
__device__ void aca_add_round_key(aca_word_t *state, aca_word_t *key);

#endif	/* AU9CUDA_AES_H */

