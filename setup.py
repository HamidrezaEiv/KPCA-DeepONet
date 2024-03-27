from setuptools import setup, find_packages

setup(
    name='kpca_deeponet',
    version='0.1',
    packages=find_packages(),
    author='Hamidreza Eivazi',
    author_email='he76@tu-clausthal.de',
    description='Nonlinear model reduction for operator learning',
    long_description=README,
    long_description_content_type='text/x-rst',
    url='https://github.com/HamidrezaEiv/KPCA-DeepONet',
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)