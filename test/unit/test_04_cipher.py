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

