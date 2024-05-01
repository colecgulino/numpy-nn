"""Sets up the package."""
from setuptools import setup, find_packages

__version__ = '0.1.0'


setup(
    name='numpy-nn-cg',
    version=__version__,
    description='Personal neural network library written in numpy.',
    author='Cole Gulino',
    author_email='cole.gulino@gmail.com',
    python_requires='>=3.11',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26',
        'torch>=2.1.2',
        'torchvision>=0.16.2'
    ],
    url='https://github.com/colecgulino/numpy-nn',
)
