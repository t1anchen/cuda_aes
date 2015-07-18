#include "key_expansion.h"
#include "utils.h"
#include "aes.h"
#include "aes.tab"

/* Import dependencies */
extern void my_cp_print_hexbytes(uint32_t *bytes, size_t bytes_len);

/* Implementation */
void aca_key_expansion(uint32_t *key, size_t key_len, uint32_t *W, size_t Nk, size_t Nr)
{
  uint i, j, cols, temp, tmp[4];
  cols = (Nr + 1) << 2;

  memcpy(W, key, (key_len >> 3)*sizeof(uint));

  for (i = Nk; i < cols; i++) {
    for (j = 0; j < 4; j++)
      tmp[j] = GET(W, j, i-1);

    if (Nk > 6) {
      if (i % Nk == 0) {
        temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
        tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
        tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
        tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
        tmp[3] = temp;
      } else if (i % Nk == 4) {
        tmp[0] = hsbox[tmp[0]];
        tmp[1] = hsbox[tmp[1]];
        tmp[2] = hsbox[tmp[2]];
        tmp[3] = hsbox[tmp[3]];
      }
    } else {
      if (i % Nk == 0) {
        temp   = hsbox[tmp[0]] ^  (Rcon[i/Nk] & 0x000000ff);
        tmp[0] = hsbox[tmp[1]] ^ ((Rcon[i/Nk] & 0xff000000) >> 24);
        tmp[1] = hsbox[tmp[2]] ^ ((Rcon[i/Nk] & 0x00ff0000) >> 16);
        tmp[2] = hsbox[tmp[3]] ^ ((Rcon[i/Nk] & 0x0000ff00) >>  8);
        tmp[3] = temp;
      }
    }
    
    for (j = 0; j < 4; j++)
      GET(W, j, i) = GET(W, j, i-Nk) ^ tmp[j];
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

