from setuptools import setup, Extension

rawref_module = Extension(
    '_rawref',
    sources=['rawrefmodule.c'],
)

setup(
    name='rawref',
    version='1.0',
    description='C extension providing mutable reference class Ref[T]',
    ext_modules=[rawref_module],
)
