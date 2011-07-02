#include "aes.h"

void my_cp_print_hexbytes(aca_word_t *bytes, aca_size_t bytes_len)
{
  aca_size_t i;
  for(i = 0; i < bytes_len; i++){
    printf("%02x", bytes[i]);
    if(!(i%16))
      printf("\n");
  }
  printf("\n");
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
