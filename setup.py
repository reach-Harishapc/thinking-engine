"""
Thinking Engine - Transparent Cognitive AI Framework
A revolutionary AI framework with biological learning mechanisms
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="thinking-engine",
    version="1.0.0",
    author="Harisha P C",
    author_email="reach.harishapc@gmail.com",
    description="Transparent Cognitive AI Framework with Biological Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reach-Harishapc/thinking-engine",
    project_urls={
        "Bug Tracker": "https://github.com/reach-Harishapc/thinking-engine/issues",
        "Documentation": "https://github.com/reach-Harishapc/thinking-engine/tree/main/docs",
        "Source Code": "https://github.com/reach-Harishapc/thinking-engine",
        "Research Paper": "https://github.com/reach-Harishapc/thinking-engine/blob/main/arxiv_submission/arxiv_paper.pdf",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai, cognitive, transparent, biological-learning, multi-agent, neural-network",
    packages=find_packages(exclude=["tests", "docs", "arxiv_submission", "*.tests", "*.tests.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "pdf": [
            "PyPDF2>=1.26",
        ],
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "thinking-engine=thinking_engine.cli:main",
            "thinking-engine-server=thinking_engine.server:main",
            "thinking-engine-test=run_multiplatform_tests:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
