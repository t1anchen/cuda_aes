import base64
import ctypes
from ctypes import *
from struct import *
from nose.tools import assert_equals

import cuda_aes_for_py

def test_my_print_hexbytes():
    src_data_struct = Struct('8I')
    src_data_buffer = ctypes.create_string_buffer(src_data_struct.size)
    values = (0x12345678, 0x90abcdef, 0, 0, 0, 0, 0, 0)
    src_data_struct.pack_into(src_data_buffer, 0, *values)
    # input_func = cuda_aes_for_py.my_print_hexbytes
    # input_func.argtypes = [c_void_p, c_uint]
    # actual = input_func(byref(src_data_buffer), sizeof(src_data_buffer)/sizeof(c_uint))
    byt = bytearray(b'\x12\x34\x56\x78\x90\xab\xcd\xef')
    actual = cuda_aes_for_py.my_print_hexbytes(byt, 2)
    assert_equals(expected, actual)

