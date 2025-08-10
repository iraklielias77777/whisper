#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read long description from README
def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read version from package
def read_version():
    here = os.path.abspath(os.path.dirname(__file__))
    version_path = os.path.join(here, 'userwhisperer', '__version__.py')
    version_dict = {}
    
    if os.path.exists(version_path):
        with open(version_path, 'r') as f:
            exec(f.read(), version_dict)
        return version_dict.get('__version__', '1.0.0')
    
    return '1.0.0'

setup(
    name='userwhisperer',
    version=read_version(),
    description='Python SDK for User Whisperer platform',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='User Whisperer Team',
    author_email='support@userwhisperer.ai',
    url='https://github.com/userwhisperer/sdk-python',
    project_urls={
        'Homepage': 'https://userwhisperer.ai',
        'Documentation': 'https://docs.userwhisperer.ai',
        'Source': 'https://github.com/userwhisperer/sdk-python',
        'Tracker': 'https://github.com/userwhisperer/sdk-python/issues',
    },
    packages=find_packages(exclude=['tests', 'tests.*']),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'requests>=2.25.0',
        'urllib3>=1.26.0,<2.0.0',
        'python-dateutil>=2.8.0',
    ],
    extras_require={
        'async': [
            'aiohttp>=3.8.0',
            'asyncio-throttle>=1.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.20.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=1.0.0',
            'isort>=5.10.0',
            'pre-commit>=2.20.0',
        ],
        'test': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.20.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'responses>=0.23.0',
            'aioresponses>=0.7.0',
        ]
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Office/Business',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords=[
        'analytics', 'user-engagement', 'personalization', 
        'machine-learning', 'customer-retention', 'behavioral-analysis'
    ],
    license='MIT',
    zip_safe=False,
)
