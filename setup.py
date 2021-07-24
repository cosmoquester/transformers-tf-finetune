from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="transformers-tf-finetune",
    version="0.0.1",
    description="Script to train hugginface transformers models with Tensorflow 2",
    python_requires=">=3.6",
    install_requires=["tensorflow>=2", "tensorflow-addons", "transformers"],
    url="https://github.com/cosmoquester/transformers-tf-finetune.git",
    author="Park Sangjun",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests"]),
)
