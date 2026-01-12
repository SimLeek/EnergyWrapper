from setuptools import setup, find_packages

with open("README.MD", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="energywrap",
    version="0.1.0",
    author="SimLeek",
    author_email="simulator.leek@example.com",
    description="wrap neural networks in an energy system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simleek/energywrapper",
    packages=[
        "energywrap",
    ],
    package_dir={"": "."},
    package_data={},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch",
    ],
)