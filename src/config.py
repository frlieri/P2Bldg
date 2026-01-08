"""
src/config.py

Configuration module to define important directory paths and initialize mail address needed for NEOS server access."""

import os

os.environ['NEOS_EMAIL'] = 'XXXX@gmail.com'

PATH_TO_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_WD = os.path.abspath(os.path.join(PATH_TO_SRC_DIR, os.pardir))