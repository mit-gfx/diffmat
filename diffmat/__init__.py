import sys

# A minimal version of Python 3.7 is required, where dictionaries begin to preserve the order of
# item insertion
if not sys.version_info >= (3, 7):
    raise RuntimeError('A minimum of Python version 3.7 is required')

from .core import config_logger, get_logger
from .translator import MaterialGraphTranslator

# Add missing version info (Issue #6)
__version__ = '0.2.0'

__all__ = ['MaterialGraphTranslator', 'config_logger', 'get_logger']
