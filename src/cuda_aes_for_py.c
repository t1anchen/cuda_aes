#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include "commons.h"
#include "utils.h"
#include "cipher.h"
#include "cuda_aes_for_py.h"

/* From utils.c */
extern size_t my_str2bytes(uint32_t **dst, const char *src);
extern size_t str2bytearray(void *buf, size_t buf_len, const char *src, size_t src_len);
extern void my_print_hexbytes(void *bytes, size_t bytes_len); 

/* From cipher.cu */
extern void aca_aes_encrypt(uint32_t *pt, uint32_t *key, uint32_t *ct, uint32_t keysize);
extern void aca_aes_decrypt(uint32_t *pt, uint32_t *key, uint32_t *ct, uint32_t keysize);


/* init */
size_t
init_key(void *key_buf, size_t buf_len, const char *src, size_t src_len)
{
  memset(key_buf, 0, buf_len);
  return str2bytearray(key_buf, buf_len, src, src_len);
}


int kick_off(int argc, char *argv[])
{
  uint32_t ct[16], *key, *pt, key_size, pt_size;
  char *key_buf, *key_str, *pt_str;
  char *pt_buf;

  /* initialize buffers */
  key_buf = (char *)malloc(100 * sizeof(char));
  pt_buf = (char *)malloc(100 * sizeof(char));

  /* check the number of args */
  if(argc < 2){
    fprintf(stderr, "USAGE: %s -[e|d]\n", argv[0]);
    exit(1);
  }

   /* choose mode: encryption or decryption */
  switch(argv[1][1]){
  case 'e':
    freopen("../test/enc.in", "r", stdin);
    freopen("../test/enc.out", "w", stdout);
    memset(key_buf, 0, 100);
    memset(pt_buf, 0, 100);
    scanf("%s", key_buf);
    scanf("%s", pt_buf);
    key_str = (char *)malloc(strlen(key_buf));
    memcpy(key_str, key_buf, strlen(key_buf));
    pt_str = (char *)malloc(strlen(key_buf));
    memcpy(pt_str, pt_buf, strlen(pt_buf)); 
    key_size = my_str2bytes(&key, key_str);
    pt_size = my_str2bytes(&pt, pt_str);
    if(key_size != 16 && key_size != 24 && key_size != 32){
      fprintf(stderr, "%s: key_size: %s\n", argv[0], strerror(EINVAL));
      exit(1);
    }
    if(pt_size != 16){
      fprintf(stderr, "%s: Invalid AES block size.\n", argv[0]);
      exit(1);
    }
    aca_aes_encrypt(pt, key, ct, key_size << 3);
    my_print_hexbytes(ct, 16);
    break;
  case 'd':
    freopen("../test/dec.in", "r", stdin);
    freopen("../test/dec.out", "w", stdout);
    memset(key_buf, 0, 100);
    memset(pt_buf, 0, 100);
    scanf("%s", key_buf);
    scanf("%s", pt_buf);
    key_str = (char *)malloc(strlen(key_buf));
    memcpy(key_str, key_buf, strlen(key_buf));
    pt_str = (char *)malloc(strlen(key_buf));
    memcpy(pt_str, pt_buf, strlen(pt_buf)); 
    key_size = my_str2bytes(&key, key_str);
    pt_size = my_str2bytes(&pt, pt_str);
    if(key_size != 16 && key_size != 24 && key_size != 32){
      fprintf(stderr, "%s: key_size: %s\n", argv[0], strerror(EINVAL));
      exit(1);
    }
    if(pt_size != 16){
      fprintf(stderr, "%s: Invalid AES block size.\n", argv[0]);
      exit(1);
    }
    aca_aes_decrypt(pt, key, ct, key_size << 3);
    my_print_hexbytes(ct, 16);
    break;
  default:
    return EINVAL;
  }

  /* garbage collection */
  free(key_buf);
  free(pt_buf);
  free(key_str);
  free(pt_str);
  return 0;
}
