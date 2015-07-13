import array
import os.path
import ctypes

def load_shared_library():
    my_lib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        '..',
        'src', 'cuda_aes_for_py.so')
    assert os.path.exists(my_lib_path) is True
    my_lib = ctypes.CDLL(my_lib_path)
    return my_lib

def test_my_str2bytearray():
    my_lib = load_shared_library()
    dst_len = ctypes.c_uint32(0x10)
    dst = ctypes.create_string_buffer(0x10)
    src = '000102030405060708090a0b0c0d0e0f'
    src_len = ctypes.c_uint32(len(src))
    my_lib.my_str2bytearray(dst, dst_len, src, src_len)
    assert dst.raw == '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'

    

    
