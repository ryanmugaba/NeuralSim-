from pathlib import Path

from setuptools import find_packages, setup

readme = Path(__file__).with_name("README.md")
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="neuralsim",
    version="0.1.0",
    description="Replay PhysioNet EEG motor imagery as a simulated 4-command brain-computer interface.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NeuralSim contributors",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.9",
    install_requires=[
        "mne>=1.5",
        "numpy>=1.23",
    ],
    extras_require={
        # Better accuracy via CSP + LDA. Pure-numpy decoder works without this.
        "sklearn": ["scikit-learn>=1.1"],
        "dev": ["pytest>=7"],
    },
    entry_points={
        "console_scripts": [
            "neuralsim-demo=neuralsim.demo:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="eeg bci brain-computer-interface mne motor-imagery physionet simulation",
)
