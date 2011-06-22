#include <stdio.h>
#include <stdlib.h>
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

  /* check the number of args */
  if(argc < 4){
    fprintf(stderr, "USAGE: %s -[e|d] <key> <txt>\n", argv[0]);
    exit(1);
  }

  /* check the validation of args */
  for(i = 0; i < 4; i++){
    if(argv[i] == NULL){
      fprintf(stderr, "%s: argv[%d]: %s", argv[0], i, strerror(EINVAL));
      exit(1);
    }
  }

  /* convert string to bytes */
  key_size = my_str2bytes(&key, argv[2]);
  pt_size = my_str2bytes(&pt, argv[3]);

  /* check key */
  if(key_size != 16 && key_size != 24 && key_size != 32){
    fprintf(stderr, "%s: key_size: %s\n", argv[0], strerror(EINVAL));
    exit(1);
  }
  if(pt_size != 16){
    fprintf(stderr, "%s: Invalid AES block size.\n", argv[0]);
    exit(1);
  }

  /* choose mode: encryption or decryption */
  switch(argv[1][1]){
  case 'e':
    aca_aes_encrypt(pt, key, ct, key_size << 3);
    my_print_hexbytes(ct, 16);
    break;
  case 'd':
    aca_aes_decrypt(pt, key, ct, key_size << 3);
    my_print_hexbytes(ct, 16);
    break;
  default:
    return EINVAL;
  }

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
    printf("%02X", bytes[i]);
  printf("\n");
}
