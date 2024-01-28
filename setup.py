#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'pip==21.1',
    'bump2version==0.5.11',
    'wheel==0.38.1',
    'watchdog==0.9.0',
    'tox==3.14.0',
    'coverage==4.5.4',
    'Sphinx==1.8.5',
    'twine==1.14.0',
    'pandas==1.5.3',
    'scikit-learn==1.2.1',
    'numpy==1.23.5',
    'matplotlib==3.7.0',
    'optuna==3.1.1',
    'lightgbm==3.3.5',
    'tqdm==4.64.1'
 ]

test_requirements = [ ]

setup(
    author="Lucas Carames",
    author_email='lgpcarames@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    description="Some tools to help the process of feature selection",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='cherrypick',
    name='cherrypick',
    packages=find_packages(include=['cherrypick', 'cherrypick.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/lgpcarames/cherrypick',
    version='0.2.1',
    zip_safe=False,
)
