#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include "aes.h"

#ifndef __AU9_AES_MAIN__
#define __AU9_AES_MAIN__
aca_size_t my_str2bytes(aca_word_t **dst, const char *src);
void my_print_hexbytes(aca_word_t *bytes, aca_size_t bytes_len);
#endif

int main(int argc, char *argv[])
{
  int i;
  aca_word_t ct[16], *key, *pt, key_size, pt_size;
  char key_buf[100], *key_str, *pt_str;
  char pt_buf[100];


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
    scanf("%s", &key_buf);
    scanf("%s", &pt_buf);
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
    scanf("%s", &key_buf);
    scanf("%s", &pt_buf);
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
  free(key_str);
  free(pt_str);
  return 0;
}

aca_size_t my_str2bytes(aca_word_t **dst, const char *src)
{
  aca_word_t i, len = strlen(src) >> 1;
  *dst = (aca_word_t *)malloc(len * sizeof(aca_word_t));

  for(i = 0; i < len; i++)
    sscanf(src + i*2, "%02X", *dst + i);

  return len;
}

void my_print_hexbytes(aca_word_t *bytes, aca_size_t bytes_len)
{
  aca_size_t i;
  for(i = 0; i < bytes_len; i++)
    printf("%02x", bytes[i]);
  printf("\n");
}
