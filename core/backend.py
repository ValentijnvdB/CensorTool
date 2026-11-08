"""
Chooses the censoring backend.

    python  the pure-Python implementation in this package (default, no extra install)
    cpp     the compiled censor_core module (installed separately), noticeably faster

Both have the same API. Code uses the ``cc`` proxy (``from core.backend import cc``), which resolves to the
selected backend on first use. Call select() before that, e.g. from the 'backend' key in pipeline.yml.
"""
from __future__ import annotations

import importlib
import importlib.util
import threading
from types import ModuleType
from typing import Any

BACKENDS = ('python', 'cpp')
DEFAULT_BACKEND = 'python'

_lock = threading.Lock()
_name: str | None = None
_module: ModuleType | None = None


class BackendError(RuntimeError):
    """The requested backend is unknown or not installed."""


def _load(name: str) -> ModuleType:
    if name == 'python':
        return importlib.import_module('core')
    try:
        return importlib.import_module('censor_core')
    except ImportError as e:
        raise BackendError(
            "backend 'cpp' needs the censor_core module, which is not installed. Build and install it (e.g. "
            "pip install /path/to/PyCensorCore), or set 'backend: python' in pipeline.yml.") from e


def normalize(name: Any) -> str:
    value = DEFAULT_BACKEND if name is None else str(name).strip().lower()
    if value in ('c++', 'c', 'censor_core'):
        value = 'cpp'
    if value not in BACKENDS:
        raise BackendError(f"unknown backend '{name}' (expected one of: {', '.join(BACKENDS)})")
    return value


def select(name: Any = None) -> ModuleType:
    """
    Selects (and imports) the backend. Selecting again with the same name is a no-op; switching to another
    backend once one is in use is an error, since objects of the two backends cannot be mixed.
    """
    global _name, _module
    value = normalize(name)
    with _lock:
        if _module is not None:
            if value != _name:
                raise BackendError(f"backend '{_name}' is already in use; cannot switch to '{value}'")
            return _module
        _module = _load(value)
        _name = value
        return _module


def get() -> ModuleType:
    """The selected backend; selects the default one if none was selected yet."""
    module = _module
    return module if module is not None else select(DEFAULT_BACKEND)


def name() -> str:
    """Name of the selected backend (selects the default one if none was selected yet)."""
    get()
    return _name


def is_available(backend: str) -> bool:
    """Whether a backend can be imported, without selecting it."""
    value = normalize(backend)
    if value == 'python':
        return True
    return importlib.util.find_spec('censor_core') is not None


class _BackendProxy:
    """Forwards attribute access to the selected backend module."""

    def __getattr__(self, attr: str) -> Any:
        return getattr(get(), attr)

    def __repr__(self) -> str:
        return f"<censor backend proxy: {_name or 'not selected'}>"


cc: Any = _BackendProxy()
