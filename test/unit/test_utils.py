import base64
import imp

cuda_aes_for_py = imp.load_source('cuda_aes_for_py', '../../src/cuda_aes_for_py.py')

def test_my_print_hexbytes():
    expected = '1234567890abcdef'
    src_data = map(int, base64.b16decode(expected))
    assert cuda_aes_for_py.my_print_hexbytes(src_data, len(src_data)) == expected + '\n'

