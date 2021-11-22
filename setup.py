from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import call

with open('dckit/README.md', 'r') as f:
    long_description = f.read()


class Installation(install):
    def run(self):
        call(['pip install -r requirements.txt --no-clean'], shell=True)
        install.run(self)


setuptools.setup(
    name='dckit',
    version='0.0.1',
    author='JC Liu',
    author_email='CLUE@CLUEbenchmarks.com',
    maintainer='DataCLUE',
    maintainer_email='CLUE@CLUEbenchmarks.com',
    description='Python toolkit for Data-centric Chinese Language Understanding Evaluation benchmark.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/CLUEBenchmark/DataCLUE',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'],
    install_requires=[],
    cmdclass={'install': Installation})
