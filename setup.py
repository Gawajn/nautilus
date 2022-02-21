import os
from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="calamari_ocr",
    version='0.1',
    packages=find_packages(),
    license="Apache License 2.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/Gawajn/nautilus",
    python_requires=">=3.7",
    install_requires=open("requirements.txt").read().split("\n"),
    keywords=["OCR", "optical character recognition", "ocropy", "ocropus", "kraken", "calamari"],
    data_files=[('', ["requirements.txt"])],
)