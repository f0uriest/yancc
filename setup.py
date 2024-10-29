"""Setup/build/install script for yancc."""

import os

import versioneer
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open(os.path.join(here, "requirements-dev.txt"), encoding="utf-8") as f:
    dev_requirements = f.read().splitlines()
dev_requirements = [
    foo for foo in dev_requirements if not (foo.startswith("#") or foo.startswith("-r"))
]

setup(
    name="yancc",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=("Yet Another NeoClassical Code"),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/f0uriest/yancc",
    author="Rory Conlin",
    author_email="roryconlin@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="neoclassical transport stellarator tokamak plasma",
    packages=find_packages(exclude=["docs", "tests", "local", "report"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.9",
    project_urls={
        "Issues Tracker": "https://github.com/f0uriest/yancc/issues",
        "Contributing": "https://github.com/f0uriest/yancc/blob/master/CONTRIBUTING.rst",  # noqa: E501
        "Source Code": "https://github.com/f0uriest/yancc/",
        "Documentation": "https://yancc.readthedocs.io/",
    },
)