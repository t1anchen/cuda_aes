#include "commons.h"

/* 
 * Interfaces of Cryptographics
 */
void aca_key_expansion(aca_word_t *key, aca_size_t key_len, aca_word_t *W, aca_size_t Nk, aca_size_t Nr);
void aca_inv_key_expansion(aca_word_t *key, aca_size_t key_len, aca_word_t *W, aca_size_t Nk, aca_size_t Nr);
