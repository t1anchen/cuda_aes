import base64
from ctypes import *
from nose.tools import assert_equals

import cuda_aes_for_py

def test_my_print_hexbytes():
    expected = '1234567890abcdef'
    my_seq = c_int * 8
    src_data = my_seq(*bytearray(base64.b16decode(expected.upper())))
    actual = cuda_aes_for_py.my_print_hexbytes(src_data, len(src_data))
    assert_equals(expected, actual)

