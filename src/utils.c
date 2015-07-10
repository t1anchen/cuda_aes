#include "utils.h"

void my_cp_print_hexbytes(uint32_t *bytes, aca_size_t bytes_len)
{
  aca_size_t i;
  for(i = 0; i < bytes_len; i++){
    printf("%02x", bytes[i]);
    if(!(i%16))
      printf("\n");
  }
  printf("\n");
}
aca_size_t my_str2bytes(uint32_t **dst, const char *src)
{
  uint32_t i, len = strlen(src) >> 1;
  *dst = (uint32_t *)malloc(len * sizeof(uint32_t));

  for(i = 0; i < len; i++)
    sscanf(src + i*2, "%02X", *dst + i);

  return len;
}

void 
my_print_hexbytes(void *bytes, size_t bytes_len)
{
  size_t i = 0;
  uint32_t *int_buf_ptr = (uint32_t *)bytes;
  for(i = 0; i < bytes_len; i++)
    printf("%02x", int_buf_ptr[i]);
  printf("\n");
}

void
print_helloworld()
{
  printf("Hello, World\n");
}
