#include "aes.h"
#include "aes.cu"


aca_errno_t aca_key_expansion(aca_byte_t *key, const aca_word_t *w, aca_word_t Nk)
{
  aca_word_t temp;

  aca_size_t i = 0;
  aca_byte_t *kp = key;

  /* Fill w from key buffer */
  while(i < Nk){
    w[i] = ACA_GETWORD(kp);
    kp+=4;
    i++;
  }

  i = Nk;

  while(i < Nb * (Nr+1)){
    temp = w[i-1];
    if(i % Nk == 0)
      temp = aca_sub_word(aca_rot_word(temp)) ^ Rcon[i / Nk];
    else
      temp = aca_sub_word(temp);
    w[i] = w[i-Nk] ^ temp;
    i++;
  }

  return NORMAL;
}
