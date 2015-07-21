import array
import hashlib
import json
import os.path
import ctypes
from ctypes import *
import utils
import struct

logger = utils.get_logger('test_03_key_expansion')

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

def test_aca_key_expansion_128():
    Nb = c_uint(4)
    key = create_string_buffer('\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c')
    key_len = c_uint32(128)
    Nk = c_uint32(4)
    Nr = c_uint32(10)
    big_w = create_string_buffer(44)
    logger.debug(map(hex, struct.unpack('>11I', big_w.raw)))
    my_lib.aca_key_expansion(key, key_len, big_w, Nk, Nr)
    logger.debug(map(hex, struct.unpack('>11I', big_w.raw)))
