import array
import hashlib
import json
import os.path
import ctypes
from ctypes import *
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
    my_lib.str2bytearray(dst, dst_len, src, src_len)
    assert dst.raw == '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'

def test_my_str2uintarray():
    dst_len = c_uint32(0x10)
    dst = create_string_buffer(0x10 * 4)
    src = '000102030405060708090a0b0c0d0e0f'
    src_len = c_uint32(len(src))
    my_lib.str2uintarray(dst, dst_len, src, src_len)
    assert dst.raw == '\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00\x07\x00\x00\x00\x08\x00\x00\x00\x09\x00\x00\x00\x0a\x00\x00\x00\x0b\x00\x00\x00\x0c\x00\x00\x00\x0d\x00\x00\x00\x0e\x00\x00\x00\x0f\x00\x00\x00'

def test_parse_test_data():
    test_data = utils.read_rsp_file('CBC', 'GFSbox', 128)
    expected = {'DECRYPT': [{'COUNT': 0, 'PLAINTEXT': 'f34481ec3cc627bacd5dc3fb08f273e6', 'CIPHERTEXT': '0336763e966d92595a567cc9ce537f5e', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 1, 'PLAINTEXT': '9798c4640bad75c7c3227db910174e72', 'CIPHERTEXT': 'a9a1631bf4996954ebc093957b234589', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 2, 'PLAINTEXT': '96ab5c2ff612d9dfaae8c31f30c42168', 'CIPHERTEXT': 'ff4f8391a6a40ca5b25d23bedd44a597', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 3, 'PLAINTEXT': '6a118a874519e64e9963798a503f1d35', 'CIPHERTEXT': 'dc43be40be0e53712f7e2bf5ca707209', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 4, 'PLAINTEXT': 'cb9fceec81286ca3e989bd979b0cb284', 'CIPHERTEXT': '92beedab1895a94faa69b632e5cc47ce', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 5, 'PLAINTEXT': 'b26aeb1874e47ca8358ff22378f09144', 'CIPHERTEXT': '459264f4798f6a78bacb89c15ed3d601', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 6, 'PLAINTEXT': '58c8e00b2631686d54eab84b91f0aca1', 'CIPHERTEXT': '08a4e2efec8a8e3312ca7460b9040bbf', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}], 'ENCRYPT': [{'COUNT': 0, 'PLAINTEXT': 'f34481ec3cc627bacd5dc3fb08f273e6', 'CIPHERTEXT': '0336763e966d92595a567cc9ce537f5e', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 1, 'PLAINTEXT': '9798c4640bad75c7c3227db910174e72', 'CIPHERTEXT': 'a9a1631bf4996954ebc093957b234589', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 2, 'PLAINTEXT': '96ab5c2ff612d9dfaae8c31f30c42168', 'CIPHERTEXT': 'ff4f8391a6a40ca5b25d23bedd44a597', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 3, 'PLAINTEXT': '6a118a874519e64e9963798a503f1d35', 'CIPHERTEXT': 'dc43be40be0e53712f7e2bf5ca707209', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 4, 'PLAINTEXT': 'cb9fceec81286ca3e989bd979b0cb284', 'CIPHERTEXT': '92beedab1895a94faa69b632e5cc47ce', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 5, 'PLAINTEXT': 'b26aeb1874e47ca8358ff22378f09144', 'CIPHERTEXT': '459264f4798f6a78bacb89c15ed3d601', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}, {'COUNT': 6, 'PLAINTEXT': '58c8e00b2631686d54eab84b91f0aca1', 'CIPHERTEXT': '08a4e2efec8a8e3312ca7460b9040bbf', 'KEY': '00000000000000000000000000000000', 'IV': '00000000000000000000000000000000'}]}
    expected_str_val = json.dumps(expected, sort_keys=True)
    expected_hash_val = hashlib.sha1(expected_str_val).hexdigest()
    actual_str_val = json.dumps(utils.parse_rsp_str(test_data), sort_keys=True)
    actual_hash_val = hashlib.sha1(actual_str_val).hexdigest()
    assert expected_hash_val == actual_hash_val
