"""Setup configuration for braintop - 3D Topology Visualizer for LLMs."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="braintop",
    version="0.1.0",
    author="Jesus Vilela",
    description="A beautiful rich 3D topology visualizer for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jesusvilela/braintop",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "networkx>=2.5",
        "scipy>=1.7.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "gguf>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "braintop=braintop.cli:main",
        ],
    },
)
