# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='causal_effect_bandits',
    version='1.0',
    description='Code reproducing the experiments of the paper'
    'Malek, Chiappa. "Asymptotically Best Causal Effect Identification '
    'with Multi-Armed Bandits", NeurIPS 2021.',
    author='DeepMind',
    author_email='alanmalek@deepmind.com',
    license='Apache License, Version 2.0',
    url='https://github.com/deepmind/abcei_mab',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'numpy',
        'matplotlib',
        'scipy',
        'sklearn',
        'typing_extensions',
    ],
    tests_require=['mock'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
