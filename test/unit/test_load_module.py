import os.path
import ctypes

def test_load_shared_library():
    my_lib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        '..',
        'src', 'cuda_aes_for_py.so')
    assert os.path.exists(my_lib_path) is True
    my_lib = ctypes.CDLL(my_lib_path)
    expected = 53
    actual = my_lib.aca_aes_print_helloworld()
    assert expected == actual

    
