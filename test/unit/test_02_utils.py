import array
import os.path
import ctypes
from ctypes import byref
from ctypes import CDLL
from ctypes import create_string_buffer
from ctypes import c_uint32
from ctypes import pointer
from ctypes import sizeof
import utils

logger = utils.get_logger('test_02_utils')

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

def test_my_str2bytearray():
    dst_len = c_uint32(0x10)
    dst = create_string_buffer(0x10)
    src = '000102030405060708090a0b0c0d0e0f'
    src_len = c_uint32(len(src))
    my_lib.my_str2bytearray(dst, dst_len, src, src_len)
    assert dst.raw == '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'

def test_my_str2bytes():
    dst = ctypes.c_buffer(0x10*4)
    src = create_string_buffer('000102030405060708090a0b0c0d0e0f')
    logger.debug(type(dst))
    logger.debug(type(src))
    my_lib.my_str2bytes(pointer(pointer(dst)), src)
    logger.debug(repr(dst.raw))
