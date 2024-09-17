from setuptools import setup, find_packages

setup(
    name="perplexity_correlations",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy"],
    extras_require={
        "dev": ["flake8>=6.0.0", "pre-commit", "black", "pytest>=7.0.0"],
    },
    author="Tristan Thrush",
    author_email="tthrush@stanford.edu",
    description="Simple and scalable tools for great pretraining data selection.",
    url="https://github.com/TristanThrush/perplexity-correlations",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
