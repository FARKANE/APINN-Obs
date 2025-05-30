#!/usr/bin/env python3
"""
Setup script for the PINN Observer package.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pinn-observer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Physics-Informed Neural Network Observer for Nonlinear Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pinn_observer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pinn-train=examples.train_modified_academic:main",
            "pinn-plot=examples.plot_modified_academic:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="physics-informed neural networks, observer design, state estimation, nonlinear systems, deep learning",
    project_urls={
        "Bug Reports": "https://github.com/FARKANE/pinn_observer/issues",
        "Source": "https://github.com/FARKANE/pinn_observer",
        "Documentation": "https://github.com/FARKANE/pinn_observer#readme",
    },
)