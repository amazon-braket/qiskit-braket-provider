"""Setup file for Qiskit-Braket provider."""

import pathlib

import setuptools

long_description = pathlib.Path("README.md").read_text(encoding="utf-8")

install_requires = pathlib.Path("requirements.txt").read_text(encoding="utf-8").splitlines()
test_requires = pathlib.Path("requirements-dev.txt").read_text(encoding="utf-8").splitlines()

with pathlib.Path("qiskit_braket_provider/_version.py").open(encoding="utf-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

setuptools.setup(
    name="qiskit_braket_provider",
    description="Qiskit-Braket provider to execute Qiskit programs on devices via Amazon Braket.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="qiskit braket sdk quantum",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={"test": test_requires},
    python_requires=">=3.11",
    version=version,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
