# Perplexity Correlations
A simple and scalable paradigm for great pretraining data selection.

## Setup

Via PIP:

```bash
pip install perplexity-correlations
```

From source:

```bash
git clone https://github.com/TristanThrush/perplexity-correlations.git
cd perplexity-correlations
pip install -e .
```

## Overview

This package can be used to get great pretraining data sampling distributions.

The input that you must provide is a matrix of (per-LLM, per-text) bits-per-byte
values, and a per-LLM vector of benchmark errors (1-accuracy) on a benchmark
that you care about. The vector of benchmark errors could be the average from
many benchmarks if you want.

The output that this package produces is a sampling distribution over pieces of
text that you could use directly to pretrain a strong LLM. Or you could use this
sampling distribution to train a fastText pretraining data filter that generalizes
to new pieces of text (reccomended).

<div align="center">
  <img src="./assets/perplexity_correlations_diagram.png" alt="Perplexity Correlations diagram" width="400"/>
</div>

Essentially, our approach encourages the training on domains where lower loss is
very correlated with higher downstream performance. We want
to pretrain on these domains, because pretraining on them will lower loss on them,
which we would expect to increase downstream performance. There is some deeper
theory here: https://arxiv.org/abs/2409.05816. Please cite this paper if you
find this package useful!

## Getting bits-per-byte from LLM loss, on various pieces of text

TODO

## Estimating optimal per-text weights

TODO

## Projecting the estimate to be a sampling distribution for pretraining

TODO

## Training a fastText pretraining data filter (optional but reccomended)

TODO


## Development Guidelines

Install the dev requirements and pre-commit hooks:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Formatting and Linting

This project uses [Black](https://black.readthedocs.io/en/stable/) for code formatting
and [Flake8](https://flake8.pycqa.org/en/latest/) for linting. After installing the
pre-commit hooks above, these will run every time you make a commit.

### Testing

If new estimation and projection functions are added, the proofs may be nontrivial and
so it is useful to test that they actually do what we think. This project uses
[pytest](https://docs.pytest.org/en/stable/) for running tests, which are also run as
GitHub actions. Just run `pytest` locally to see if your tests pass.

## Citation

```bibtex
@misc{thrush2024perplexitycorrelations,
      title={Improving Pretraining Data Using Perplexity Correlations}, 
      author={Tristan Thrush and Christopher Potts and Tatsunori Hashimoto},
      year={2024},
      eprint={2409.05816},p
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.05816}, 
}
```


