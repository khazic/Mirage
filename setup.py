from setuptools import setup, find_packages

# Production dependencies
INSTALL_REQUIRES = [
    "openai>=1.0.0",
    "aiohttp>=3.8.0",
    "tqdm>=4.65.0",
    "jsonlines>=3.1.0",
    "click>=8.0.0",
]

# Development dependencies
EXTRA_REQUIRES = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "mypy>=1.5.0",
        "ruff>=0.1.0",
        "pre-commit>=3.5.0",
    ],
    "test": [
        "pytest>=7.0.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "sphinx-autodoc-typehints>=1.24.0",
    ],
}

setup(
    name="mmoment",
    version="0.1.0",
    description="Multi-Modal Modeling and Evaluation on Novel Tasks",
    author="Your Name",
    author_email="khazzz1c@gmail.com",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRES,
    python_requires=">=3.11",
    entry_points={
        'console_scripts': [
            'mmoment-eval=mmoment.cli:evaluate',
            'mmoment-test=mmoment.cli:test_api'
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 