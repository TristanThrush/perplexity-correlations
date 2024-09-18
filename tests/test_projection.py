import numpy as np
import pytest
from perplexity_correlations.projection import linear, l2

tau = np.array([0.2, 0.5, 0.9, 0.4, 1])
estimate = np.array([-0.3, 4, 1, 0.9, 0])


def test_linear():
    solution = np.array([0, 0.5, 0.5, 0, 0])
    assert linear(estimate, tau) == pytest.approx(solution, abs=1e-12)


def test_l2():
    solution = np.array([0, 0.5, 0.3, 0.2, 0])
    assert l2(estimate, tau) == pytest.approx(solution, abs=1e-12)
