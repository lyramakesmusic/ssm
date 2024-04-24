from setuptools import setup, find_packages

setup(
    name="ssm",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "librosa",
        "numpy",
        "matplotlib",
    ],
)
