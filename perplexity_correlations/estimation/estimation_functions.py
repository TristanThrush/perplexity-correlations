import numpy as np
import warnings


def _value_check_X_and_y(X, y):
    if np.any(X < 0):
        warnings.warn(
            "X has negative entries,\
are you sure this is a matrix of bits-per-byte values?"
        )
    if np.any((y < 0) | (y > 1)):
        warnings.warn(
            "y has negative outside of the\
range [0,1], are you sure this is a vector of benchmark errors?"
        )
    if X.ndim != 2:
        raise ValueError(f"X has {X.ndim} dimensions but expected 2.")
    if y.ndim != 1:
        raise ValueError(f"y has {y.ndim} dimensions but expected 1.")


def product(X, y):
    """
    In expectation, this function returns a vector proportional to
    the optimal weight vector relating the per-LLM benchmark error
    vector (y), and the per-LLM and per-text bits-per-byte matrix (X).
    In addition to standard high-dimensional regression assumptions,
    we assume the relationship between X and y is:

    y = f(<w,X> + e),

    where w is the vector of optimal per-text weights that we want to
    estimate, e is zero-mean error, and f is a monotonically increasing
    function which we do not have to know.

    This function uses the single-index model parameter estimator from
    Plan et al. (2016): https://arxiv.org/abs/1404.3749, which is the
    U-statistic:

    x_k*y_k,

    for 1<=k<=N where N is the number of LLMs, x_k is the per-text
    bit-per-byte vector of the k-th LLM, and y_k is the benchmark
    error of the k-th LLM.

    NOTE: This estimator is not robust to outliers in X or y.

    Parameters
    ----------
    X : numpy.ndarray
        A NxD matrix with bits-per-byte values of N LLMs on D pieces of text.
    y : numpy.array
        A N-length vector with the benchmark error (1-accuracy) of the N LLMs.

    Returns
    -------
    numpy.array
        The D-length vector estimate of w.

    Raises
    ------
    ValueError
        If X is not 2 dimensional.
    ValueError
        If y is not 1 dimensional.


    Examples
    --------
    >>> import numpy as np
    >>> from perplexity_correlations.estimation import product
    >>>
    >>> # Bits-per-byte from 100 LLMs on 20000 text domains:
    >>> X = np.random.rand(100, 20000)
    >>>
    >>> # Benchmark error from the 100 LLMs:
    >>> y = np.random.uniform(low=0, high=1, size=(100))
    >>>
    >>> # Estimate the weights for the relationship:
    >>> estimate = product(X, y)
    """
    _value_check_X_and_y(X, y)
    estimate = np.mean((y.T * X.T), axis=1)
    return estimate


def sign(X, y):
    """
    In expectation, this function returns a vector proportional to
    the optimal weight vector relating the per-LLM benchmark error
    vector (y), and the per-LLM and per-text bits-per-byte matrix (X).
    In addition to standard high-dimensional regression assumptions,
    we assume the relationship between X and y is:

    y = f(<w,X> + e),

    where w is the vector of optimal per-text weights that we want to
    estimate, e is zero-mean error, and f is a monotonically increasing
    function which we do not have to know.

    This function uses the single-index model parameter estimator from
    Chen & Banerjee (2017): https://proceedings.mlr.press/v70/chen17a.html,
    which is the U-statistic:

    sign(y_g-y_k)*(x_g-x_k),

    for 1<=k,g<=N where N is the number of LLMs, x_k is the per-text
    bit-per-byte vector of the k-th LLM, and y_k is the benchmark
    error of the k-th LLM.

    NOTE: This estimator is not robust to outliers in X,
    but is robust to outliers in y.

    Parameters
    ----------
    X : numpy.ndarray
        A NxD matrix with bits-per-byte values of N LLMs on D pieces of text.
    y : numpy.array
        A N-length vector with the benchmark error (1-accuracy) of the N LLMs.

    Returns
    -------
    numpy.array
        The D-length vector estimate of w.

    Raises
    ------
    ValueError
        If X is not 2 dimensional.
    ValueError
        If y is not 1 dimensional.


    Examples
    --------
    >>> import numpy as np
    >>> from perplexity_correlations.estimation import sign
    >>>
    >>> # Bits-per-byte from 100 LLMs on 20000 text domains:
    >>> X = np.random.rand(100, 20000)
    >>>
    >>> # Benchmark error from the 100 LLMs:
    >>> y = np.random.uniform(low=0, high=1, size=(100))
    >>>
    >>> # Estimate the weights for the relationship:
    >>> estimate = sign(X, y)
    """
    _value_check_X_and_y(X, y)

    estimate_sum = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        estimate_sum += np.sum(
            np.sign(y[i:] - y[i])[:, np.newaxis] * (X[i:] - X[i]),
            axis=0,
        )

    estimate = 2 * estimate_sum / (X.shape[0] * (X.shape[0] - 1))
    return estimate


def sign_cdf(X, y):
    """
    In expectation, this function returns a vector of values with the same
    relative ranks as the values in the optimal weight vector relating
    the per-LLM benchmark error vector (y), and the per-LLM and per-text
    bits-per-byte matrix (X). In addition to standard high-dimensional
    regression assumptions, we assume the relationship between X and y is:

    y = f(<w,X> + e),

    where w is the vector of optimal per-text weights that we want to
    estimate, e is zero-mean error, and f is a monotonically increasing
    function which we do not have to know.

    This function uses the single-index model parameter estimator from
    Thrush et al. (2024): https://arxiv.org/abs/2409.05816,
    which is the U-statistic:

    sign(y_g-y_k)*(CDF(x_g)-CDF(x_k)),

    for 1<=k,g<=N where N is the number of LLMs, x_k is the per-text
    bit-per-byte vector of the k-th LLM, y_k is the benchmark
    error of the k-th LLM, and CDF computes the column-wise empirical CDF of
    the entries in the x vectors.

    NOTE: This estimator is robust to outliers in X and y.

    Parameters
    ----------
    X : numpy.ndarray
        A NxD matrix with bits-per-byte values of N LLMs on D pieces of text.
    y : numpy.array
        A N-length vector with the benchmark error (1-accuracy) of the N LLMs.

    Returns
    -------
    numpy.array
        The D-length vector estimate of w.

    Raises
    ------
    ValueError
        If X is not 2 dimensional.
    ValueError
        If y is not 1 dimensional.


    Examples
    --------
    >>> import numpy as np
    >>> from perplexity_correlations.estimation import sign_cdf
    >>>
    >>> # Bits-per-byte from 100 LLMs on 20000 text domains:
    >>> X = np.random.rand(100, 20000)
    >>>
    >>> # Benchmark error from the 100 LLMs:
    >>> y = np.random.uniform(low=0, high=1, size=(100))
    >>>
    >>> # Estimate the weights for the relationship:
    >>> estimate = sign_cdf(X, y)
    """
    _value_check_X_and_y(X, y)

    X_ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1

    X_cdf = X_ranks / X.shape[0]

    estimate_sum = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        estimate_sum += np.sum(
            np.sign(y[i:] - y[i])[:, np.newaxis] * (X_cdf[i:] - X_cdf[i]),
            axis=0,
        )

    estimate = 2 * estimate_sum / (X.shape[0] * (X.shape[0] - 1))
    return estimate


def sign_sign(X, y):
    """
    In expectation, this function returns a vector of values with the same
    relative ranks as the values in the optimal weight vector relating
    the per-LLM benchmark error vector (y), and the per-LLM and per-text
    bits-per-byte matrix (X). In addition to standard high-dimensional
    regression assumptions, we assume the relationship between X and y is:

    y = f(<w,X> + e),

    where w is the vector of optimal per-text weights that we want to
    estimate, e is zero-mean error, and f is a monotonically increasing
    function which we do not have to know.

    This function uses the single-index model parameter estimator from
    Thrush et al.'s (2024) (https://arxiv.org/abs/2409.05816) initial experiments,
    although the preprint does not document it yet. It is the U-statistic:

    sign(y_g-y_k)*sign(x_g-x_k),

    for 1<=k,g<=N where N is the number of LLMs, x_k is the per-text
    bit-per-byte vector of the k-th LLM, y_k is the benchmark
    error of the k-th LLM.

    NOTE: This estimator is robust to outliers in X and y.

    Parameters
    ----------
    X : numpy.ndarray
        A NxD matrix with bits-per-byte values of N LLMs on D pieces of text.
    y : numpy.array
        A N-length vector with the benchmark error (1-accuracy) of the N LLMs.

    Returns
    -------
    numpy.array
        The D-length vector estimate of w.

    Raises
    ------
    ValueError
        If X is not 2 dimensional.
    ValueError
        If y is not 1 dimensional.


    Examples
    --------
    >>> import numpy as np
    >>> from perplexity_correlations.estimation import sign_sign
    >>>
    >>> # Bits-per-byte from 100 LLMs on 20000 text domains:
    >>> X = np.random.rand(100, 20000)
    >>>
    >>> # Benchmark error from the 100 LLMs:
    >>> y = np.random.uniform(low=0, high=1, size=(100))
    >>>
    >>> # Estimate the weights for the relationship:
    >>> estimate = sign_sign(X, y)
    """
    _value_check_X_and_y(X, y)

    estimate_sum = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        estimate_sum += np.sum(
            np.sign(y[i:] - y[i])[:, np.newaxis] * np.sign(X[i:] - X[i]),
            axis=0,
        )

    estimate = 2 * estimate_sum / (X.shape[0] * (X.shape[0] - 1))
    return estimate


def spearmanr(X, y):
    """
    In expectation, this function returns a vector of values with the same
    relative ranks as the values in the optimal weight vector relating
    the per-LLM benchmark error vector (y), and the per-LLM and per-text
    bits-per-byte matrix (X). In addition to standard high-dimensional
    regression assumptions, we assume the relationship between X and y is:

    y = f(<w,X> + e),

    where w is the vector of optimal per-text weights that we want to
    estimate, e is zero-mean error, and f is a monotonically increasing
    function which we do not have to know.

    This function uses the single-index model parameter estimator from
    Thrush et al. (2024): https://arxiv.org/abs/2409.05816,
    which is the elementwise Spearman rank correlation, following:

    1-6*(sum_{k=1}^N(Rank(y_k)-Rank(x_k))^2)/(N*(N^2-1)),

    where N is the number of LLMs, x_k is the per-text
    bit-per-byte vector of the k-th LLM, and y_k is the benchmark
    error of the k-th LLM.

    NOTE: This estimator is robust to outliers in X and y.

    NOTE: The current version of the Thrush et al. paper does not provide the proof
    that this estimator matches the ranks of the optimal weights in expectation,
    but we have now proved this.

    Parameters
    ----------
    X : numpy.ndarray
        A NxD matrix with bits-per-byte values of N LLMs on D pieces of text.
    y : numpy.array
        A N-length vector with the benchmark error (1-accuracy) of the N LLMs.

    Returns
    -------
    numpy.array
        The D-length vector estimate of w.

    Raises
    ------
    ValueError
        If X is not 2 dimensional.
    ValueError
        If y is not 1 dimensional.


    Examples
    --------
    >>> import numpy as np
    >>> from perplexity_correlations.estimation import spearmanr
    >>>
    >>> # Bits-per-byte from 100 LLMs on 20000 text domains:
    >>> X = np.random.rand(100, 20000)
    >>>
    >>> # Benchmark error from the 100 LLMs:
    >>> y = np.random.uniform(low=0, high=1, size=(100))
    >>>
    >>> # Estimate the weights for the relationship:
    >>> estimate = spearmanr(X, y)
    """
    _value_check_X_and_y(X, y)

    X_ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1

    y_ranks = np.argsort(np.argsort(y, axis=0), axis=0) + 1

    estimate = 1 - 6 * np.sum((y_ranks.T - X_ranks.T) ** 2, axis=1) / (
        X.shape[0] * (X.shape[0] ** 2 - 1)
    )
    return estimate
