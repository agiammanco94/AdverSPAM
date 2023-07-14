#! /usr/bin/env python

import os
import sys
from setuptools import setup, find_packages, dist


DISTNAME = 'AdverSPAM'
DESCRIPTION = "Correlations-aware Adversarial Attack for Spam Account in OSNs"
AUTHOR = 'Andrea Giammanco'
AUTHOR_EMAIL = 'andrea.giammanco@unipa.it'
URL = 'https://github.com/agiammanco94/AdverSPAM'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/agiammanco94/AdverSPAM'
VERSION = '0.0.0'

# Install setup requirements
dist.Distribution().fetch_build_eggs(['Cython', 'numpy', 'scipy'])

try:
    import numpy
except ImportError:
    print('numpy is required for installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required for installation')
    sys.exit(1)

try:
    from Cython.Build import cythonize
except ImportError:
    print('Cython is required for installation')
    sys.exit(1)


if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    with open('requirements.txt') as f:
        INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

    setup(
        name=DISTNAME,
        packages=find_packages(),
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        install_requires=INSTALL_REQUIRES,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        zip_safe=False,
    )