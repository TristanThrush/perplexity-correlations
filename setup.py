from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="perplexity_correlations",
    version="0.1.2",
    packages=find_packages(),
    install_requires=["numpy"],
    extras_require={
        "dev": [
            "flake8>=6.0.0",
            "pre-commit",
            "black",
            "pytest>=7.0.0",
            "sphinx",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
            "setuptools",
            "wheel",
            "twine",
        ],
    },
    author="Tristan Thrush",
    author_email="tthrush@stanford.edu",
    description="Simple and scalable tools for great pretraining data selection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TristanThrush/perplexity-correlations",
    keywords="AI, artificial intelligence, machine learning, deep learning, research,\
language models, LLM, pretraining data, data selection, training data, statistics,\
stats ML",
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        # License
        "License :: OSI Approved :: MIT License",
        # Operating System
        "Operating System :: OS Independent",
        # Programming Language
        "Programming Language :: Python :: 3",
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    project_urls={
        "Documentation": "https://tristanthrush.github.io/perplexity-correlations/",
        "Source": "https://github.com/TristanThrush/perplexity-correlations",
        "Tracker": "https://github.com/TristanThrush/perplexity-correlations/issues",
    },
)
