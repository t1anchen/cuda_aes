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
#endif /* ACA_TYPE */

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
#endif /* ACA_BENCHMARK */


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
#endif /* ACA_ERRNO */

#endif	/* AU9CUDA_AES_H */

