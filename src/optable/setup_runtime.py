from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np


class SetupScriptRuntime:
    """Execute an experiment setup script with stable path semantics."""

    def __init__(self, setup_file: os.PathLike[str] | str):
        self.setup_path = Path(setup_file).expanduser().resolve()
        if not self.setup_path.exists():
            raise FileNotFoundError(f"Setup file not found: {self.setup_path}")
        if not self.setup_path.is_file():
            raise ValueError(f"Setup path is not a file: {self.setup_path}")

        self.setup_dir = self.setup_path.parent
        self.source_code = self.setup_path.read_text(encoding="utf-8")
        self._compiled_code = compile(
            self.source_code, str(self.setup_path), mode="exec"
        )

    @contextmanager
    def _temporary_cwd(self) -> Iterator[None]:
        prev_cwd = os.getcwd()
        os.chdir(self.setup_dir)
        try:
            yield
        finally:
            os.chdir(prev_cwd)

    @contextmanager
    def _patched_numpy_load(self) -> Iterator[None]:
        original_load = np.load
        setup_dir = self.setup_dir

        def _load_with_setup_dir(file: Any, *args: Any, **kwargs: Any):
            if isinstance(file, (str, os.PathLike)):
                path = Path(file)
                if not path.is_absolute():
                    file = str((setup_dir / path).resolve())
            return original_load(file, *args, **kwargs)

        np.load = _load_with_setup_dir  # type: ignore[assignment]
        try:
            yield
        finally:
            np.load = original_load  # type: ignore[assignment]

    def execute(self, namespace: Dict[str, Any]) -> None:
        namespace.setdefault("__builtins__", __builtins__)
        namespace["__file__"] = str(self.setup_path)
        namespace["__name__"] = "__optable_setup__"
        namespace["__package__"] = None

        with self._temporary_cwd():
            with self._patched_numpy_load():
                exec(self._compiled_code, namespace)
