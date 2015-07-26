import array
import hashlib
import json
import os.path
import ctypes
from ctypes import *
import utils
import struct

logger = utils.get_logger('test_04_cipher')

def load_shared_library():
    my_lib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        '..',
        'src', 'cuda_aes_for_py.so')
    assert os.path.exists(my_lib_path) is True
    my_lib = CDLL(my_lib_path)
    return my_lib

my_lib = load_shared_library()

def test_aca_cipher_encrypt_128():
    Nb = c_uint(4)              # Nb = 4
    cipher_key = create_string_buffer('\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c')
    plain_input = create_string_buffer('\x32\x43\xf6\xa8\x88\x5a\x30\x8d\x31\x31\x98\xa2\xe0\x37\x07\x34')
    encrypted_output = create_string_buffer(4*4) # 4 word
    expected = '\x39\x02\xdc\x19\x25\xdc\x11\x6a\x84\x09\x85\x0b\x1d\xfb\x97\x32'
    # my_lib.aca_aes_encrypt(plain_input, cipher_key, encrypted_output, 128)
    # actual = struct.unpack('4I', encrypted_output.raw)
    # assert actual == expected

def test_aca_aes_128():
    Nk = 4
    Nr = 10
    plain_input = create_string_buffer('\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff')
    key_input = create_string_buffer('\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f')
    encrypted_output = create_string_buffer('\x69\xc4\xe0\xd8\x6a\x7b\x04\x30\xd8\xcd\xb7\x80\x70\xb4\xc5\x5a')

def test_aca_aes_192():
    Nk = 6
    Nr = 12
    plain_input = create_string_buffer('\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff')
    key_input = create_string_buffer('\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17')
    encrypted_output = create_string_buffer('\xdd\xa9\x7c\xa4\x86\x4c\xdf\xe0\x6e\xaf\x70\xa0\xec\x0d\x71\x91')

def test_aca_aes_256():
    Nk = 8
    Nr = 14
    plain_input = create_string_buffer('\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff')
    key_input = create_string_buffer('\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f')
    encrypted_output = create_string_buffer('\x8e\xa2\xb7\xca\x51\x67\x45\xbf\xea\xfc\x49\x90\x4b\x49\x60\x89')

