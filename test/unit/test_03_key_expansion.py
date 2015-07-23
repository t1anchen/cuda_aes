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
    big_w = create_string_buffer(44 * 4)
    expected = (0x2b7e1516, 0x28aed2a6, 0xabf71588, 0x9cf4f3c, 0xa0fafe17, 0x88542cb1, 0x23a33939, 0x2a6c7605, 0xf2c295f2, 0x7a96b943, 0x5935807a, 0x7359f67f, 0x3d80477d, 0x4716fe3e, 0x1e237e44, 0x6d7a883b, 0xef44a541, 0xa8525b7f, 0xb671253b, 0xdb0bad00, 0xd4d1c6f8, 0x7c839d87, 0xcaf2b8bc, 0x11f915bc, 0x6d88a37a, 0x110b3efd, 0xdbf98641, 0xca0093fd, 0x4e54f70e, 0x5f5fc9f3, 0x84a64fb2, 0x4ea6dc4f, 0xead27321, 0xb58dbad2, 0x312bf560, 0x7f8d292f, 0xac7766f3, 0x19fadc21, 0x28d12941, 0x575c006e, 0xd014f9a8, 0xc9ee2589, 0xe13f0cc8, 0xb6630ca6)
    my_lib.aca_key_expansion(key, key_len, big_w, Nk, Nr)
    actual = struct.unpack('44I', big_w.raw)
    assert actual == expected

def test_aca_key_expansion_192():
    Nb = c_uint(4)              # Nb = 4
    key = create_string_buffer('\x8e\x73\xb0\xf7\xda\x0e\x64\x52\xc8\x10\xf3\x2b\x80\x90\x79\xe5\x62\xf8\xea\xd2\x52\x2c\x6b\x7b')
    key_len = c_uint32(192)     # key_len = 128, 192, 256
    Nk = c_uint32(6)            # Nk = 4, 6, 8
    Nr = c_uint32(12)           # Nr = Nk + 6
    big_w = create_string_buffer(52 * 4) # word w[Nb*(Nr+1)]
    expected = (0x8e73b0f7, 0xda0e6452, 0xc810f32b, 0x809079e5, 0x62f8ead2, 0x522c6b7b, 0xfe0c91f7, 0x2402f5a5, 0xec12068e, 0x6c827f6b, 0xe7a95b9, 0x5c56fec2, 0x4db7b4bd, 0x69b54118, 0x85a74796, 0xe92538fd, 0xe75fad44, 0xbb095386, 0x485af057, 0x21efb14f, 0xa448f6d9, 0x4d6dce24, 0xaa326360, 0x113b30e6, 0xa25e7ed5, 0x83b1cf9a, 0x27f93943, 0x6a94f767, 0xc0a69407, 0xd19da4e1, 0xec1786eb, 0x6fa64971, 0x485f7032, 0x22cb8755, 0xe26d1352, 0x33f0b7b3, 0x40beeb28, 0x2f18a259, 0x6747d26b, 0x458c553e, 0xa7e1466c, 0x9411f1df, 0x821f750a, 0xad07d753, 0xca400538, 0x8fcc5006, 0x282d166a, 0xbc3ce7b5, 0xe98ba06f, 0x448c773c, 0x8ecc7204, 0x1002202)
    my_lib.aca_key_expansion(key, key_len, big_w, Nk, Nr)
    actual = struct.unpack('52I', big_w.raw)
    assert actual == expected
    
    
