# Perplexity Correlations

This package can be used to get LLM pretraining data sampling distributions using
simple statistical methods that are on par with the state of the art in our tests.
The compute requirements are minimal. You don't need to train any LLMs
yourself.

Essentially, our approach encourages training on domains where lower loss is
very correlated with higher downstream performance. We can use existing and freely available LLMs to do this:

<div align="center">
  <img src="https://raw.githubusercontent.com/TristanThrush/perplexity-correlations/main/assets/perplexity_correlations_diagram.png" alt="Perplexity Correlations diagram" width="400"/>
</div>

We want to pretrain on these correlated domains, because pretraining on them will
lower loss on them, which we would expect to increase downstream performance. There
is some deeper theory here: https://arxiv.org/abs/2409.05816. Please cite this paper
if you find this package useful!

The input that you must provide is a matrix of (per-LLM, per-text) bits-per-byte
values, and a per-LLM vector of benchmark errors (ranging from 0 to 1, with lower
meaning better) on a benchmark that you care about. The vector of benchmark errors
could be the average from many benchmarks if you want.

The output that our methods produce is a sampling distribution over the texts,
and you could use it directly to pretrain a strong LLM. Or you could use this
sampling distribution to train a fastText pretraining data filter that generalizes
to new pieces of text (reccomended).

Note that you can use a heterogenous set of LLMs to get the pretraining sampling
distribution: they can have different tokenizers, architectures, scales, and
pretraining data. Another essential feature here is that the number of texts can be
far larger than the number of LLMs; this package uses very high-dimensional regression
methods.

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

## Getting bits-per-byte from LLM loss, on various pieces of text

Our approach requires you to generate an input matrix of (per-LLM, per-text)
bits-per-byte (BPB) values. BPB normalizes loss to reduce the impact of tokenizer
differences (see: https://arxiv.org/abs/2101.00027). To get bits-per-byte
from an LLM, you just need to do this (assuming the loss is the causal language
modelling loss averaged over tokens, which is typical):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

text = "Some text from some web domain that I could use for pretraining."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs['input_ids']

outputs = model(input_ids, labels=input_ids)

loss = outputs.loss.item()
num_tokens = input_ids.shape[-1]
num_bytes = len(text.encode('utf-8'))

bits_per_byte = (num_tokens/num_bytes)*loss/np.log(2)
```

## Estimating optimal per-text weights

Once you have the bits-per-byte values from many LLMs on a bunch of texts, you
can organize these values into a NxD numpy ndarray (N=number of LLMs, D=number of
texts). You also need to get a N-length numpy array of LLM benchmark errors on some
benchmark (or average of benchmarks). These errors should range from 0 to 1, and lower
should mean better.

With these numpy arrays, you can use a variety of estimators that our package provides
to estimate the optimal weights relating performance and text losses (assuming
an unknown monotonic relationship between performance and loss, which is more general
than typical power-law or sigmoid scaling laws). We recommend using one of the following
options:

```python
from perplexity_correlations.estimation import spearmanr

estimate = spearmanr(bits_per_byte_matrix, benchmark_error_vector)
```

or

```python
from perplexity_correlations.estimation import sign_cdf

estimate = sign_cdf(bits_per_byte_matrix, benchmark_error_vector)
```

Note that these particular estimators are robust to outliers in the data, but the cost
is that they only return estimates for the optimal weights that we can trust
up to the ranks. In other words, the estimate might be [-0.1, 0.2, 0.35, 0.9] where
the true optimal weights are [-0.1, 0.28, 0.5, 1.1]. We will see below that the ranks
alone can still be used to get a nice pretraining sampling distribution.


## Projecting the estimate to be a sampling distribution for pretraining

We now have an estimate for a vector with weights that correspond with the ranks of
the optimal weight vector. But we still need to project it so that it is a sampling
distribution that we could use for pretraining. Obviously, it needs to satisfy the
constraint that the elements should be non-negative and sum to 1. But also, we don't
want our algorithm to tell you to train on 300 billion tokens of Wikipedia if you only
have 3 billion tokens, so we should also have the sampling distribution satisfy a
per-text constraint that prevents the weights from being so high that you will have to
duplicate data from any text domains. The following code projects our estimate to
satisfy these constraints, where `tau` is the vector of per-domain thresholds:

```python
from perplexity_correlations.projection import linear

projected_estimate = linear(estimate, tau)
```

In our paper, we choose the `tau` values to be as high as possible per-domain such that
we won't be duplicating data. This works when we have more pretraining data than
we can train on, but you might want to choose other reasonable thresholds if
you are using our technique to upsample pretraining data instead of filter it.

Finally, it turns out that the `perplexity_correlations.projection.linear` has the
nice property of only depending on the ranks of the values in `estimate`, so we only
need to pass in an estimate for the optimal weights that is trustworthy up to the
ranks.

We provide another speedy projection method called
`perplexity_correlations.projection.l2`, which does depend on the values. It is
best used with estimators such as `perplexity_correlations.estimation.product` and 
`perplexity_correlations.estimation.sign` which return estimates that are proportional
to the optimal weights in expectation, but we found that these estimators are not
robust enough to be particularly useful. Still, they are nice to have for further
research.


## Training a fastText pretraining data filter

Now, we have a sampling distribution that we could use for pretraining a LLM, but only
on the text domains that we actually included in our estimate. How are we going to
scale this approach a bit better? The linear projection above has the nice property of
making the weight for the i-th domain either 0 or the max possible value (don't include
it at all or include all of it). We can treat these include/don't include judgements as
labels for each text:

```python
labels = []
for weight in projected_estimate:
    labels.append("include" if weight > 0 else "exclude")
```

Now we can make a text file where each line is formatted as:

```python
f'__label__{label} {text}'
```

Then, we can train a [fastText](https://pypi.org/project/fasttext/) classifier:

```python
import fasttext

# Train the FastText model
model = fasttext.train_supervised(input='train.txt', wordNgrams=2)

# Evaluate the model
result = model.test('test.txt')
print(f'Number of samples: {result[0]}')
print(f'Precision@1: {result[1]}')
print(f'Recall@1: {result[2]}')

# Save the model
model.save_model('fasttext_filter.bin')
```

Then, we can apply this fastText model as a filter, choosing pretraining
data that gets the highest `'include'` scores until we reach our pretraining
token budget.


## Full API documentation

https://tristanthrush.github.io/perplexity-correlations/


## Development guidelines

Install the dev requirements and pre-commit hooks:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Formatting and linting

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
      eprint={2409.05816},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.05816}, 
}
```


