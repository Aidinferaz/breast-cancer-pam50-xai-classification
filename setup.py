import os
from setuptools import setup, find_packages

def read_requirements():
    requirements = []
    if os.path.isfile('requirements.txt'):
        with open('requirements.txt') as f:
            requirements = f.read().splitlines()
    return requirements


setup(
    name="cancer_xai_lib",
    version="0.1.0",
    description="Library XAI untuk analisis kanker payudara",
    author="Nama Anda",

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    install_requires=read_requirements(),

    python_requires=">=3.8",
)