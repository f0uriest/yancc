"""Setup/build/install script for yancc.

Package metadata lives in ``pyproject.toml``. This file exists only because
versioneer needs to inject the version and its custom build commands, which
cannot be expressed declaratively.
"""

import os
import sys

from setuptools import setup

# PEP 517 backends do not put the source root on sys.path, so the vendored
# versioneer.py next to this file would not otherwise be importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import versioneer  # noqa: E402

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
