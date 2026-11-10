import constants

SERVER_DIR = constants.data_root / "server"
"""Every file the server keeps, apart from the detection cache. Unused files are removed (see storage.py)."""

UPLOAD_DIR = SERVER_DIR / "uncensored"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CENSORED_PATH = SERVER_DIR / "censored"
CENSORED_PATH.mkdir(parents=True, exist_ok=True)

DEBUG_DIR = SERVER_DIR / "debug"

CACHE_DIR = constants.cache_root / 'server'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
