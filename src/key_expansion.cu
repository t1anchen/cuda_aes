#include "key_expansion.h"
#include "utils.h"
#include "aes.h"
#include "aes.tab"
#include <arpa/inet.h>          /* for htonl and ntohl */

/* Import dependencies */
extern void my_cp_print_hexbytes(uint32_t *bytes, size_t bytes_len);

/* Export interface */
/* extern "C" void aca_key_expansion(uint32_t *key, size_t key_len, uint32_t *W, size_t Nk, size_t Nr); */

/* Implementation */

/* RotWord(word(a0, a1, a2, a3)) -> word(a1, a2, a3, a0) */
static uint32_t RotWord(uint32_t src)
{
  return (uint32_t)(((src & 0x00ffffff) << 8) | ((src & 0xff000000) >> 24));
}

/* SubWord needs BigEndian word */
static uint32_t SubWord(uint32_t src)
{
  uint32_t dst = 0;
  dst |= ((hsbox[(src & 0xff000000) >> 24] << 24) & 0xff000000);
  dst |= ((hsbox[(src & 0x00ff0000) >> 16] << 16) & 0x00ff0000);
  dst |= ((hsbox[(src & 0x0000ff00) >>  8] <<  8) & 0x0000ff00);
  dst |= ((hsbox[(src & 0x000000ff)      ]      ) & 0x000000ff);
  return dst;
}

void aca_key_expansion(void *key_buf, size_t key_len, void *W_buf, size_t Nk, size_t Nr)
{
  size_t i, j, cols;
  uint32_t temp, tmp[4];
  uint32_t *key = (uint32_t *)key_buf;
  uint32_t *W = (uint32_t *)W_buf;
  cols = (Nr + 1) << 2;

  for (i = 0; i < Nk; i++) {
    W[i] = htonl(key[i]);
  }


  for (i = Nk; i < cols; i++) {
    temp = W[i-1];
    if (i % Nk == 0)
      temp = SubWord(RotWord(temp)) ^ Rcon[i/Nk];
    else if ((Nk > 6) && (i % Nk == 4))
      temp = SubWord(temp);
    W[i] = W[i-Nk] ^ temp;

    /* for (j = 0; j < 4; j++) */
    /*   tmp[j] = GET(W, j, i-1) & 0xff; */

    /* if (Nk > 6) { */
    /*   if (i % Nk == 0) { */
    /*     temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff); */
    /*     tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24); */
    /*     tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16); */
    /*     tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8); */
    /*     tmp[3] = temp; */
    /*   } else if (i % Nk == 4) { */
    /*     tmp[0] = hsbox[tmp[0]]; */
    /*     tmp[1] = hsbox[tmp[1]]; */
    /*     tmp[2] = hsbox[tmp[2]]; */
    /*     tmp[3] = hsbox[tmp[3]]; */
    /*   } */
    /* } else { */
    /*   if (i % Nk == 0) { */
    /*     temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff); */
    /*     tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24); */
    /*     tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16); */
    /*     tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8); */
    /*     tmp[3] = temp; */
    /*   } */
    /* } */
    
    /* for (j = 0; j < 4; j++)  */
    /*   fprintf(stderr, "i = %u, tmp[%u] = %02p\n", i, j, tmp[j]); */
    /* for (j = 0; j < 4; j++) */
    /*   GET(W, j, i) = (uint32_t)(GET(W, j, i-Nk) ^ tmp[j]); */
  }

}

void aca_inv_key_expansion(uint32_t *key, size_t key_len, uint32_t *W, size_t Nk, size_t Nr)
{
  uint i, j, cols, temp, tmp[4];
  uint32_t *cW_tmp;
  size_t cW_tmp_size;

  cols = (Nr + 1) << 2;

  memcpy(W, key, (key_len >> 3)*sizeof(uint));

  for(i=Nk; i<cols; i++) {
    for(j=0; j<4; j++)
      tmp[j] = GET(W, j, i-1);
    if(Nk > 6) {
      if(i % Nk == 0) {
	temp   = inv_hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
	tmp[0] = inv_hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
	tmp[1] = inv_hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
	tmp[2] = inv_hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
	tmp[3] = temp;
      } else if(i % Nk == 4) {
	tmp[0] = inv_hsbox[tmp[0]];
	tmp[1] = inv_hsbox[tmp[1]];
	tmp[2] = inv_hsbox[tmp[2]];
	tmp[3] = inv_hsbox[tmp[3]];
      }
    } else {
      if(i % Nk == 0) {
	temp   = inv_hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
	tmp[0] = inv_hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
	tmp[1] = inv_hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
	tmp[2] = inv_hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
	tmp[3] = temp;
      }
    }
    for(j=0; j<4; j++)
      GET(W, j, i) = GET(W, j, i-Nk) ^ tmp[j];
  }

  /* IMPORTANT:
   * W should be handled in GPU memory, 
   * so cW_tmp should be created here to
   * be as a temperary copy in GPU 
   */
  cW_tmp_size = ((Nr + 1) * sizeof(uint32_t)) << 2;
  cudaMalloc((void**)&cW_tmp, cW_tmp_size);
  cudaMemcpy(cW_tmp, W, cW_tmp_size, cudaMemcpyHostToDevice);

  for(i = 1; i < Nr; i++)
    aca_inv_mix_columns<<<1,(cols>>2)>>>(cW_tmp);

  memset(W, 0, cW_tmp_size);
  cudaMemcpy(W, cW_tmp, cW_tmp_size, cudaMemcpyDeviceToHost);

  /* If debug flag defined,
   * it will output the contents of W
   */
  /* my_cp_print_hexbytes(W, cW_tmp_size); */
}

