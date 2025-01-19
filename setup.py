from setuptools import setup, find_packages

setup(
    name="Mmoment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "aiohttp",
        "tqdm",
        "jsonlines"
    ],
) 