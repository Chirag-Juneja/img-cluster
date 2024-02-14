from setuptools import find_packages, setup

setup(
    name="pinaka",
    version="0.0.1",
    description="Advanced Computer Vision Library",
    package_dir={"": "pinaka"},
    packages=find_packages(where="pinaka"),
    author="Chirag Juneja",
    author_email="chiragjuneja6@gmail.com",
)
