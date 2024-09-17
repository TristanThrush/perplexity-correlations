import numpy as np
import pytest
from perplexity_correlations.estimation import (
    # product,
    # sign,
    sign_cdf,
    sign_sign,
    # spearmanr,
)

# We test with X~N(0, I), and noise~N(0, SIGMA^2),
# following standard high-dimensional regression assumptions.
NUM_SAMPLES = 10000
DIM = 10
SIGMA = 0.5

X = np.random.randn(NUM_SAMPLES, DIM)
noise = np.random.randn(NUM_SAMPLES) * SIGMA

non_negative_weights = np.array(list(range(DIM)))  # weights looks like [0,1,2,3,...]
non_positive_weights = -np.array(
    list(range(DIM))
)  # weights looks like [0,-1,-2,-3,...]
alternating_sign_weights = np.array(
    [(-num if num % 2 == 0 else num) for num in list(range(DIM))]
)  # weights looks like [0,1,-2,3,-4...]
more_peaky_alternating_sign_weights = np.array(
    [(-(num**2) if num % 2 == 0 else num**2) for num in list(range(DIM))]
)  # weights looks like [0,1,-4,9,-16...]
some_weights = []
for weights in [
    non_negative_weights,
    non_positive_weights,
    alternating_sign_weights,
    more_peaky_alternating_sign_weights,
]:
    # Normalize weight vectors. Proofs assume WLOG that ||weights||_2 = 1.
    # See estimator references.
    some_weights.append(weights / np.sqrt(np.sum(weights**2)))


def linear_f(x):
    return x


def sigmoid_f(x):
    return 2 / (1 + np.exp(-x)) - 1


def exponential_f(x):
    return 3**x


some_fs = [linear_f, sigmoid_f, exponential_f]


def test_product():
    # TODO
    pass


def test_sign():
    # TODO
    pass


def test_sign_cdf():
    for weights in some_weights:
        for f in some_fs:
            y = f(X @ weights + noise)
            estimate = sign_cdf(X, y)

            # The proof says that the estimate equals this:
            assert (2 / np.pi) * np.arcsin(
                weights / (2 * np.sqrt(1 + SIGMA**2))
            ) == pytest.approx(estimate, abs=2e-2)


def test_sign_sign():
    for weights in some_weights:
        for f in some_fs:
            y = f(X @ weights + noise)
            estimate = sign_sign(X, y)

            # The proof says that the estimate equals this:
            assert (2 / np.pi) * np.arcsin(
                weights / (np.sqrt(1 + SIGMA**2))
            ) == pytest.approx(estimate, abs=2e-2)


def test_spearmanr():
    # TODO
    pass
