import numpy as np
import warnings


def _value_check_estimate_and_tau(estimate, tau):
    if estimate.ndim != 1:
        raise ValueError(f"estimate has {estimate.ndim} dimensions but expected 1.")
    if tau.ndim != 1:
        raise ValueError(f"tau has {tau.ndim} dimensions but expected 1.")
    if np.any(tau <= 0):
        raise ValueError(
            "tau values must be positive.\
If you want certain tau values to be zero, then\
run the estimation without those domains."
        )
    if np.sum(tau) < 1:
        raise ValueError("Projection is infeasible because sum of tau values is < 1.")


def linear(estimate, tau):
    """
    Given the estimate from one of the estimator methods, this method projects
    it, maximizing the dot product (linear projection) subject to:

    sum(projected_estimate) = 1
    0 <= projected_estimate[i] <= tau[i]

    It uses the fast projection solution from Thrush et al. (2024):
    https://arxiv.org/abs/2409.05816

    This projection turns the estimate into a sampling distribution that you could use
    for training a model on D different domains of text
    (where len(estimate) == len(tau) == D). tau specifies constraints that prevent you
    from upsampling a domain of text too much. In Thrush et al., the standard choice
    for tau[i] is to set it as large as possible such that you won't duplicate data by
    sampling the i-th domain with weight tau[i].

    NOTE: the solution here is not dependent upon the exact values in the estimate;
    it only depends on their ranks. This makes it easy to directly use the estimates
    from estimation.sign_cdf, estimation.sign_sign, and estimation.spearmanr,
    which compute strictly monotonically increasing trig functions of the optimal
    weights in expectation. Read more at https://arxiv.org/abs/2409.05816.

    Parameters
    ----------
    estimate : numpy.darray
        A D-length vector returned from one of the perplexity_correlations.estimation
        methods.
    tau : numpy.array
        A D-length vector with the per-domain sampling thresholds.

    Returns
    -------
    numpy.array
        The D-length projected estimate to be used as a pretraining sampling
        distribution.

    Raises
    ------
    ValueError
        If values in tau sum to less than 1.
    ValueError
        If any values in tau are non-positive.
    ValueError
        If estimate is not 1 dimensional.
    ValueError
        If tau is not 1 dimensional.


    Examples
    --------
    >>> import numpy as np
    >>> from perplexity_correlations.estimation import spearmanr
    >>> from perplexity_correlations.projection import linear
    >>>
    >>> # Bits-per-byte from 100 LLMs on 20000 text domains:
    >>> X = np.random.rand(100, 20000)
    >>>
    >>> # Benchmark error from the 100 LLMs:
    >>> y = np.random.uniform(low=0, high=1, size=(100))
    >>>
    >>> # Estimate the weights for the relationship:
    >>> estimate = spearmanr(X, y)
    >>>
    >>> # per-domain sampling thresholds
    >>> # (the sum of this will almost certainly be >= 1)
    >>> tau = np.random.rand(20000)
    >>>
    >>> projected_estimate = linear(estimate, tau)
    """

    _value_check_estimate_and_tau(estimate, tau)

    indices_of_largest_to_smallest = np.argsort(-1 * estimate)
    tau_sort = tau[indices_of_largest_to_smallest]
    projected_estimate = np.zeros_like(estimate, dtype=np.float32)
    tau_cum_sum = np.cumsum(tau_sort)
    find_index = np.min(np.nonzero(tau_cum_sum >= 1)[0])
    if find_index == 0:
        projected_estimate[indices_of_largest_to_smallest[0]] = 1
    else:
        projected_estimate[indices_of_largest_to_smallest[:find_index]] = tau_sort[
            :find_index
        ]
        if (1 - np.sum(projected_estimate)) > tau_sort[find_index]:
            warnings.warn(
                f"numerical issues likely caused slight violation of\
tau bounds: {1-np.sum(projected_estimate)} > {tau_sort[find_index]}"
            )
        projected_estimate[indices_of_largest_to_smallest[find_index]] = 1 - np.sum(
            projected_estimate
        )
    return projected_estimate


def l2(estimate, tau, atol=1e-12):
    """
    Given the estimate from one of the estimator methods, this method projects
    it, minimizing the L_2 norm subject to:

    sum(projected_estimate) = 1
    0 <= projected_estimate[i] <= tau[i]

    It uses the fast projection solution from Thrush et al. (2024):
    https://arxiv.org/abs/2409.05816

    This projection turns the estimate into a sampling distribution that you could use
    for training a model on D different domains of text
    (where len(estimate) == len(tau) == D). tau specifies constraints that prevent you
    from upsampling a domain of text too much. In Thrush et al., the standard choice
    for tau[i] is to set it as large as possible such that you won't duplicate data by
    sampling the i-th domain with weight tau[i].

    NOTE: unlike projection.linear, the solution here is dependent upon the exact
    values in the estimate, not just their ranks. To use this projection on estimates
    from estimation.sign_cdf, estimation.sign_sign, and estimation.spearmanr, you must
    invert the monotonic trig functions from the solutions to uncover the exact values
    of the weight estimates, and potentially learn the norm through hyperparameter
    search. Even after doing this, it is unlikely that you will be able to uncover the
    true weight values if your LLM bits-per-byte data deviates too much from the
    Gaussian distribution. Read more at https://arxiv.org/abs/2409.05816.

    Parameters
    ----------
    estimate : numpy.darray
        A D-length vector returned from one of the perplexity_correlations.estimation
        methods (or monotonically transformed estimate if using one of the robust
        estimators from Thrush et al.).
    tau : numpy.array
        A D-length vector with the per-domain sampling thresholds.
    atol : float, optional
        Allowable margin of error for each projected weight. Smaller values will make
        the bisection search take longer. Default is 1e-12.


    Returns
    -------
    numpy.array
        The D-length projected estimate to be used as a pretraining sampling
        distribution.

    Raises
    ------
    ValueError
        If values in tau sum to less than 1.
    ValueError
        If any values in tau are non-positive.
    ValueError
        If estimate is not 1 dimensional.
    ValueError
        If tau is not 1 dimensional.


    Examples
    --------
    >>> import numpy as np
    >>> from perplexity_correlations.estimation import sign
    >>> from perplexity_correlations.projection import l2
    >>>
    >>> # Bits-per-byte from 100 LLMs on 20000 text domains:
    >>> X = np.random.rand(100, 20000)
    >>>
    >>> # Benchmark error from the 100 LLMs:
    >>> y = np.random.uniform(low=0, high=1, size=(100))
    >>>
    >>> # Estimate the weights for the relationship:
    >>> estimate = sign(X, y)
    >>>
    >>> # per-domain sampling thresholds
    >>> # (the sum of this will almost certainly be >= 1)
    >>> tau = np.random.rand(20000)
    >>>
    >>> projected_estimate = l2(estimate, tau)
    """

    _value_check_estimate_and_tau(estimate, tau)

    # Semi-fast projection that does bisection search to find the intersection point.

    # Solves the projection problem using the up-and-down strategy.
    # Find an offset c such that clip(estimate+c, 0, tau) sums to 1.
    offset_min = -np.max(estimate)
    offset_max = np.max(tau - estimate)

    # Now do a bisection search to find the optimal offset.
    while offset_max - offset_min > atol:
        offset = (offset_min + offset_max) / 2
        projected_estimate = np.clip(estimate + offset, 0, tau)
        if np.sum(projected_estimate) > 1:
            offset_max = offset
        else:
            offset_min = offset
    return projected_estimate
