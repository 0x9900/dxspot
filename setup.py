#!/usr/bin/env python3
# Fred (W6BSD) 2022
#
import sys

from setuptools import setup

import dxspots

__author__ = "Fred C. (W6BSD)"
__version__ = dxspots.__version__
__license__ = 'BSD'

py_version = sys.version_info[:2]
if py_version < (3, 5):
  raise RuntimeError('dxspots requires Python 3.5 or later')


def readme():
  with open('README.md', encoding="utf-8") as fdr:
    return fdr.read()


setup(
  name='dxspots',
  version=__version__,
  description='DXCC spot statistics',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/0x9900/dxspot/',
  license=__license__,
  author=__author__,
  author_email='w6bsd@bsdworld.org',
  py_modules=['dxspots'],
  install_requires=['matplotlib', 'numpy', 'scipy'],
  entry_points={
    'console_scripts': ['dxspots = dxspots:main'],
  },
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Telecommunications Industry',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Topic :: Communications :: Ham Radio',
  ],
)
