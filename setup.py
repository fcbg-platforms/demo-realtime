from pathlib import Path
from setuptools import setup, find_packages


# Version
version = None
with open(Path(__file__).parent/'template'/'_version.py', 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break

if version is None:
    raise RuntimeError('Could not determine version.')

# Descriptions
short_description = """Short template package description"""
long_description_file = Path(__file__).parent / 'README.md'
with open(long_description_file, 'r') as file:
    long_description = file.read()
if long_description_file.suffix == '.md':
    long_description_content_type='text/markdown'
elif long_description_file.suffix == '.rst':
    long_description_content_type='text/x-rst'
else:
    long_description_content_type='text/plain'

# Variables
NAME = 'template'
DESCRIPTION = short_description
LONG_DESCRIPTION = long_description
LONG_DESCRIPTION_CONTENT_TYPE=long_description_content_type
AUTHOR = 'Mathieu Scheltienne'
AUTHOR_EMAIL = 'mathieu.scheltienne@gmail.com'
MAINTAINER = 'Mathieu Scheltienne'
MAINTAINER_EMAIL = 'mathieu.scheltienne@gmail.com'
URL = 'https://github.com/mscheltienne/template-python'
LICENSE = 'MIT License'
DOWNLOAD_URL = 'https://github.com/mscheltienne/template-python'
VERSION = version


# Dependencies
def get_requirements(path):
    """Get mandatory dependencies from file."""
    install_requires = list()
    with open(path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            req = line.strip()
            if len(line) == 0:
                continue
            install_requires.append(req)

    return install_requires

install_requires = get_requirements('requirements.txt')


setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3 :: Only',
        'Natural Language :: English'
        ],
    keywords='neuroscience neuroimaging',
    project_urls={
        'Documentation': 'https://github.com/mscheltienne/template-python',
        'Source': 'https://github.com/mscheltienne/template-python',
        'Tracker': 'https://github.com/mscheltienne/template-python/issues'
        },
    platforms='any',
    python_requires='>=3.7',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    )
