import setuptools
import os

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setuptools.setup(
    name="GeoPrivacy",
    version="0.0.1",
    author="Ao Qu",
    author_email="ao.qu@vanderbilt.edu",
    description="A python package for implementing geo-indistinguishability privacy mechanisms",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/quao627/GeoPrivacy",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["GeoPrivacy"],
    include_package_data=True,
    install_requires=[
        "geopy", "numpy", "networkx", "gurobipy", "scipy"
    ],
)
