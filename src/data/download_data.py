from itertools import zip_longest

import tqdm as tqdm
from typing import Dict, List
from urllib3 import PoolManager
from pathlib import Path
import logging
from tqdm import tqdm
from hashlib import md5
import pickle

PROCESSED_NAMES_PICKLE = Path('processed_names.pickle')

RAW_FILE = Path('raw_names.txt')


def preprocess_raw() -> Dict[str, str]:
    # if PROCESSED_NAMES_PICKLE.exists():
    #     return pickle.load(PROCESSED_NAMES_PICKLE.open('rb'))
    # if not RAW_FILE.exists():
    #     raise Exception(f'`{RAW_FILE}` does not exist! You must create it.')
    with RAW_FILE.open() as f:
        data = f.read().split()
        structured_data: Dict[str, str] = {}
        for i in range(0, len(data), 4):
            if data[0].endswith('image_crop.tif'):
                structured_data.update({data[i]: data[i + 1][4:]})
        with PROCESSED_NAMES_PICKLE.open('wb') as g:
            g.write(pickle.dumps(structured_data))
    return structured_data


ARGS_FILE = Path('args.pickle')


def create_args(structured_data: Dict[str, str]) -> Dict[str, object]:
    if not ARGS_FILE.exists():
        arguments = {
            'urls': list(
                map(lambda x: f'https://zenodo.org/record/4746605/files/{x}?download=1', structured_data.keys())),
            'save_dir': 'raw',
            'filenames': structured_data.keys(),
            'checksums': structured_data.values()
        }
        with PROCESSED_NAMES_PICKLE.open('wb') as g:
            g.write(pickle.dumps(structured_data))
    else:
        return pickle.load(ARGS_FILE.open('rb'))
    return arguments


def verify_checksum(filepath: str or Path = None, checksum: str = None) -> bool:
    if filepath is None:
        return False
    if checksum is None:
        return False
    else:
        try:
            bytes.fromhex(checksum)
        except ValueError as e:
            logging.log(logging.ERROR, 'Something wrong with checksum string!', e)
            return False
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if not filepath.exists():
        return False
    try:
        return md5(filepath.read_bytes()).hexdigest() == checksum
    except MemoryError as e:
        logging.log(logging.ERROR, 'Something wrong with checksum string!', e)
    return False


def download_data(urls, save_dir, filenames: List[str or Path] = None, checksums: List[str or bytes] = None,
                  verify=False):
    pm = PoolManager()
    for url, file, check in tqdm(zip_longest(urls, map(lambda x: Path(save_dir) / x, filenames), checksums),
                                 total=len(urls), unit='files', desc='Files downloaded'):
        if file.exists():
            if not verify:
                logging.log(logging.INFO, f'Operations were skipped for {file}!!!')
                continue
            elif verify_checksum(file, check):
                logging.log(logging.INFO, f'The file `{file}` has been verified!!!')
                continue

        with pm.request('GET', url, preload_content=False) as resp:
            logging.log(logging.INFO, f'Downloading and saving to {file}...')
            content_length = int(resp.headers['Content-Length'])
            download_ok = False
            while not download_ok:

                with file.open('wb') as wbf:
                    for chunk in tqdm(resp.stream(1024), total=content_length // 1024 + (content_length % 1024 == 0),
                                      unit='KiB', leave=True, position=0, desc=f'Downloading `{file.name}`',
                                      colour='green'):
                        wbf.write(chunk)

                download_ok = verify_checksum(file, check) if verify else True


if __name__ == '__main__':
    print(preprocess_raw())
    # download_data(**create_args(preprocess_raw()), verify=True)
