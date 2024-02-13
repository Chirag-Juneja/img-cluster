from setuptools import find_packages, setup

setup(
    name="imgcluster",
    version="0.0.1",
    description="Unsupervised Image Clustering module",
    package_dir={"": "imgcluster"},
    packages=find_packages(where="imgcluster"),
    author='Chirag Juneja',
    author_email='chiragjuneja6@gmail.com',
)
