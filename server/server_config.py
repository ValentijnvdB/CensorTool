import constants

UPLOAD_DIR = constants.data_root / "server/uncensored/"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CENSORED_PATH = constants.data_root / "server/censored/"
CENSORED_PATH.mkdir(parents=True, exist_ok=True)

CACHE_DIR = constants.cache_root / 'server'
CACHE_DIR.mkdir(parents=True, exist_ok=True)