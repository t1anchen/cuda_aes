import logging
import os.path
import zipfile

AES_TYPES = ['OFB', 'ECB', 'CFB', 'CFB128', 'CFB8', 'CFB1', 'CBC']
AES_PROPS = ['VarTxt', 'VarKey', 'KeySbox', 'GFSbox']
AES_BLOCK_SIZES = [128, 192, 256]

def read_rsp_file(aes_type='CBC', 
                  aes_prop='VarTxt', 
                  aes_block_size=128):
    zip_pkg_path = os.path.join('KAT_AES.zip')
    assert aes_type in AES_TYPES
    assert aes_prop in AES_PROPS
    assert aes_block_size in AES_BLOCK_SIZES
    assert os.path.exists(zip_pkg_path) is True
    zip_pkg = zipfile.ZipFile(zip_pkg_path)
    rsp_file_name = ''.join([
        aes_type, aes_prop, 
        str(aes_block_size), '.rsp'])
    assert len(filter(lambda x: x.filename == rsp_file_name, 
                      zip_pkg.filelist)) > 0
    return zip_pkg.read(rsp_file_name)

def get_logger(logger_name):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    return log


    
    
