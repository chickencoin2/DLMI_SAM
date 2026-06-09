"""BackendManager - owns the single active SamBackend and handles switching.

Only one backend is resident at a time. `switch()` tears down the old backend's
sessions/hooks, unloads it (freeing VRAM), then constructs + loads the new one,
rolling back to the previous backend if the new one fails to load. git/3.1 are
lazily loaded on first selection.
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import logging

import torch

from .base import BackendError

logger = logging.getLogger("DLMI_SAM_LABELER.Backends.Manager")

# Display labels for the 3 selectable backends.
BACKEND_LABELS = {"hug": "HuggingFace", "git": "GitHub SAM3", "3.1": "GitHub SAM3.1"}


class BackendManager:
    def __init__(self, app):
        self.app = app
        self.active = None
        self._availability_cache = None

    # ------------------------------------------------------------------ #
    # Availability probing (cached)
    # ------------------------------------------------------------------ #
    def availability(self, refresh: bool = False) -> dict:
        if self._availability_cache is not None and not refresh:
            return self._availability_cache
        avail = {"hug": False, "git": False, "3.1": False}
        # HF transformers SAM3 classes present?
        try:
            from transformers import Sam3TrackerVideoModel  # noqa: F401
            avail["hug"] = True
        except Exception as e:
            logger.info(f"HF SAM3 unavailable: {e}")
        # Official sam3 package + our GitBackend module both importable?
        git_ok = importlib.util.find_spec("sam3") is not None
        if git_ok:
            try:
                importlib.import_module("util.backends.git_backend")
            except Exception as e:
                logger.info(f"GitBackend not ready yet: {e}")
                git_ok = False
        avail["git"] = git_ok
        avail["3.1"] = git_ok
        self._availability_cache = avail
        return avail

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _construct(self, key: str):
        app = self.app
        device = app.device
        dtype = getattr(app, "model_dtype", None)
        if key == "hug":
            from .hf_backend import HFBackend
            return HFBackend(app, device, dtype)
        if key in ("git", "3.1"):
            from .git_backend import GitBackend
            version = "sam3" if key == "git" else "sam3.1"
            backend = GitBackend(app, device, dtype, version=version)
            backend.key = key
            backend.label = BACKEND_LABELS[key]
            return backend
        raise BackendError(f"Unknown backend key: {key!r}")

    # ------------------------------------------------------------------ #
    # Switching
    # ------------------------------------------------------------------ #
    def switch(self, key: str, on_status=None) -> bool:
        def status(msg):
            logger.info(msg)
            if on_status:
                try:
                    on_status(msg)
                except Exception:
                    pass

        if self.active is not None and self.active.key == key and self.active.is_loaded():
            return True

        avail = self.availability(refresh=True)
        if not avail.get(key, False):
            status(f"Backend '{key}' is unavailable.")
            return False

        prev = self.active
        try:
            if prev is not None:
                try:
                    if hasattr(self.app, "_teardown_for_backend_switch"):
                        self.app._teardown_for_backend_switch()
                except Exception as te:
                    logger.warning(f"Session teardown before switch failed: {te}")
                prev.unload()
                self.active = None
                self._empty_cache()
            backend = self._construct(key)
            status(f"Loading backend '{backend.label}'...")
            backend.load()
            self.active = backend
            status(f"Backend '{backend.label}' ready.")
            return True
        except Exception as e:
            logger.exception(f"Backend switch to '{key}' failed")
            self.active = None
            if prev is not None:
                try:
                    prev.load()
                    self.active = prev
                    status(f"Switch failed ({e}); rolled back to '{prev.label}'.")
                    return False
                except Exception:
                    logger.exception("Rollback to previous backend also failed")
            status(f"Backend '{key}' load failed: {e}")
            return False

    def _empty_cache(self):
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
