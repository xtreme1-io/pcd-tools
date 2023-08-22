# encoding=utf-8
import re
from os.path import join, dirname
from setuptools import setup, find_packages


def read_file_content(filepath):
    with open(join(dirname(__file__), filepath), encoding="utf8") as fp:
        return fp.read()


def find_version(filepath):
    content = read_file_content(filepath)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version(join("pcd_tools", "__init__.py"))
long_description = read_file_content("README.md")

setup(
    name="pcd-tools",
    version=VERSION,
    url="https://github.com/xtreme1-io/pcd-tools.git",
    author="Dasheng Ji",
    author_email="jidasheng@basicfinder.com",
    description="PCD tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tornado",
        "py-healthcheck",
        "numpy",
        "scipy",
        "requests",
        "Pillow",
        "python-lzf",
        "easydict",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "pcd2png=pcd_tools.pcd2image:main",
            "pcd2binary=pcd_tools.pcd_normalize:main",
        ]
    },
)
