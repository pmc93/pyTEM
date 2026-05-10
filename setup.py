from setuptools import setup, find_packages

setup(
    name="pytem-app",
    version="0.1",
    packages=find_packages(exclude=["app", "notebooks", "aarhus_inv"]),
)
