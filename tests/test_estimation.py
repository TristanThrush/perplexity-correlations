import numpy as np
import pytest
from perplexity_correlations.estimation import (
    product,
    sign,
    sign_cdf,
    sign_sign,
    spearmanr,
)

# We test with X~N(0, I), and noise~N(0, SIGMA^2),
# following standard high-dimensional regression assumptions.
NUM_SAMPLES = 10000
DIM = 10

some_sigmas = [0, 0.5]

X = np.random.randn(NUM_SAMPLES, DIM)

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
    for weights in some_weights:
        for f in some_fs:
            for sigma in some_sigmas:
                noise = np.random.randn(NUM_SAMPLES) * sigma
                y = f(X @ weights + noise)
                estimate = product(X, y)
                normalized_estimate = estimate / np.sqrt(np.sum(estimate**2))

                # The proof says that the estimate equals this in expectation:
                # Note that the variance with this estimator tends to be higher
                assert weights == pytest.approx(
                    normalized_estimate, abs=5e-2 if sigma == 0 else 1e-1
                )


def test_sign():
    for weights in some_weights:
        for f in some_fs:
            for sigma in some_sigmas:
                noise = np.random.randn(NUM_SAMPLES) * sigma
                y = f(X @ weights + noise)
                estimate = sign(X, y)
                normalized_estimate = estimate / np.sqrt(np.sum(estimate**2))

                # The proof says that the estimate equals this in expectation:
                assert weights == pytest.approx(
                    normalized_estimate, abs=3e-2 if sigma == 0 else 8e-2
                )


def test_sign_cdf():
    for weights in some_weights:
        for f in some_fs:
            for sigma in some_sigmas:
                noise = np.random.randn(NUM_SAMPLES) * sigma
                y = f(X @ weights + noise)
                estimate = sign_cdf(X, y)

                monotonically_transformed_weights = (2 / np.pi) * np.arcsin(
                    weights / (2 * np.sqrt(1 + sigma**2))
                )
                # The proof says that the estimate equals this in expectation:
                assert monotonically_transformed_weights == pytest.approx(
                    estimate, abs=1.5e-2 if sigma == 0 else 4e-2
                )


def test_sign_sign():
    for weights in some_weights:
        for f in some_fs:
            for sigma in some_sigmas:
                noise = np.random.randn(NUM_SAMPLES) * sigma
                y = f(X @ weights + noise)
                estimate = sign_sign(X, y)
                monotonically_transformed_weights = (2 / np.pi) * np.arcsin(
                    weights / (np.sqrt(1 + sigma**2))
                )
                # The proof says that the estimate equals this in expectation:
                assert monotonically_transformed_weights == pytest.approx(
                    estimate, abs=3e-2 if sigma == 0 else 8e-2
                )


def test_spearmanr():
    for weights in some_weights:
        for f in some_fs:
            for sigma in some_sigmas:
                # TODO: finish proof for nonzero noise
                if sigma != 0:
                    continue

                noise = np.random.randn(NUM_SAMPLES) * sigma
                y = f(X @ weights + noise)
                estimate = spearmanr(X, y)

                intermediate_transform = weights / (np.sqrt(2 - (weights**2)))
                monotonically_transformed_weights = 1 - 6 * (X.shape[0] ** 2) * (
                    1 / 6
                    - np.arctan(
                        intermediate_transform
                        / np.sqrt(intermediate_transform**2 + 2)
                    )
                    / np.pi
                ) / (X.shape[0] ** 2 - 1)

                # The proof says that the estimate equals this in expectation:
                assert monotonically_transformed_weights == pytest.approx(
                    estimate, abs=3e-2 if sigma == 0 else 8e-2
                )
