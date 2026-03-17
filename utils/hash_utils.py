import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def dictionary_hash(to_hash: dict, hash_len: int):
    return hash_bytes(json.dumps( to_hash, sort_keys=True, ensure_ascii=True).encode('utf-8'), hash_len)


def hash_bytes(to_hash: str|bytes, hash_len: int):
    dict_hash = hashlib.md5(to_hash).hexdigest()
    return dict_hash[:hash_len]


def write_object( variable: Any, filename: Path ):
    with gzip.open( filename, 'wb' ) as file:
        pickle.dump(variable, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_object( filename: Path ):
    with gzip.open( filename, 'rb' ) as file:
        return pickle.load(file)


def md5_for_file( filename: Path, length: int ):
    assert length <= 32
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[32 - length:]
