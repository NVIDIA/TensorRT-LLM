"""rawref - Mutable reference with singleton pattern.

This module provides a C extension for creating mutable references to Python
objects, similar to weakref.ref but with manual invalidation control and a
singleton pattern via __rawref__.

The main purpose is to work around the issue that mypyc does not support
weakref.

Main exports:
- ReferenceType: The reference class
- ref: Alias for ReferenceType (recommended, like weakref.ref)
- NULL: Invalid reference constant for initialization
"""

from ._rawref import NULL, ReferenceType, ref

__all__ = ["ReferenceType", "ref", "NULL"]

__version__ = "2.0.0"
