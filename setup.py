import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vl_checklist",
    packages = find_packages(),
    #packages = ["om_vlp"],
    package_data={'vl_checklist': ['*.yaml']},
    include_package_data=True,
    version="0.0.1",
    author="kyusonglee",
    description="Checklist for Vision language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/om-ai-lab",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        "tqdm",
        "Image",
        "pyyaml"
    ]
)
