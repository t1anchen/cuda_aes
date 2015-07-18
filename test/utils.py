import logging
import os.path
import re
import zipfile

AES_TYPES = ['OFB', 'ECB', 'CFB', 'CFB128', 'CFB8', 'CFB1', 'CBC']
AES_PROPS = ['VarTxt', 'VarKey', 'KeySbox', 'GFSbox']
AES_BLOCK_SIZES = [128, 192, 256]

def read_rsp_file(aes_type='CBC', 
                  aes_prop='VarTxt', 
                  aes_block_size=128):
    zip_pkg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'KAT_AES.zip')
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

def parse_rsp_str(rsp_str):
    section = ''
    subsection = -1
    result = {}

    for line in filter(lambda line: re.match(r'#.*', line) is None, rsp_str.split('\r\n')):
        if line == '[ENCRYPT]':
            section = 'ENCRYPT'
            result['ENCRYPT'] = []
            continue
        elif line == '[DECRYPT]':
            section = 'DECRYPT'
            result['DECRYPT'] = []
            continue
        if section == 'ENCRYPT':
            if line != '':
                k, v = line.split(' = ')
                if k == 'COUNT':
                    result['ENCRYPT'].append({k: int(v)})
                    subsection = int(v)
                else:
                    result['ENCRYPT'][subsection][k] = v
        if section == 'DECRYPT':
            if line != '':
                k, v = line.split(' = ')
                if k == 'COUNT':
                    result['DECRYPT'].append({k: int(v)})
                    subsection = int(v)
                else:
                    result['DECRYPT'][subsection][k] = v
    return result


def get_logger(logger_name):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    return log


    
    
