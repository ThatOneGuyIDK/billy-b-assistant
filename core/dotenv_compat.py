"""Compatibility helpers for optional python-dotenv support.

This keeps the app runnable when python-dotenv is not installed by falling
back to a small standard-library implementation for loading and updating
simple .env files.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import get_key as _get_key
    from dotenv import load_dotenv as _load_dotenv
    from dotenv import set_key as _set_key
except ModuleNotFoundError:
    _get_key = None
    _load_dotenv = None
    _set_key = None


def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None) -> bool:
    if _load_dotenv is not None:
        return bool(_load_dotenv(dotenv_path=dotenv_path))

    if dotenv_path is None:
        dotenv_path = Path(".env")

    path = Path(dotenv_path)
    if not path.exists():
        return False

    loaded = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)
        loaded = True

    return loaded


def get_key(dotenv_path: str | os.PathLike[str], key_to_get: str) -> str | None:
    if _get_key is not None:
        return _get_key(dotenv_path, key_to_get)

    path = Path(dotenv_path)
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key.strip() != key_to_get:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value

    return None


def set_key(
    dotenv_path: str | os.PathLike[str],
    key_to_set: str,
    value_to_set: str,
    quote_mode: str = "always",
) -> tuple[bool, str, str]:
    if _set_key is not None:
        return _set_key(dotenv_path, key_to_set, value_to_set, quote_mode=quote_mode)

    path = Path(dotenv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    found = False
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    new_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _ = line.split("=", 1)
            if key.strip() == key_to_set:
                new_lines.append(f"{key_to_set}={value_to_set}")
                found = True
                continue
        new_lines.append(raw_line)

    if not found:
        new_lines.append(f"{key_to_set}={value_to_set}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True, key_to_set, value_to_set