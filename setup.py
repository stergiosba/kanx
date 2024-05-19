"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="kanx",
    version="0.0.1",
    description="Fast Implementation of Kolmogorov-Arnold Networks in JAX",
    author="Stergios, Bachoumas",
    author_email="stevbach@udel.edu",
    license="Apache License, Version 2.0",
    url="https://github.com/stergiosba/kanx",
    packages=find_packages(
        exclude=["img", "notebooks", "tests", "examples"]
    ),
    install_requires=[
        "numpy",
    ],
)