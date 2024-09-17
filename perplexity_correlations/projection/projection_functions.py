import numpy as np


def _value_check_estimate_and_tau(estimate, tau):
    if estimate.ndim != 1:
        raise ValueError(f"estimate has {estimate.ndim} dimensions but expected 1.")
    if tau.ndim != 1:
        raise ValueError(f"tau has {tau.ndim} dimensions but expected 1.")
    if np.any(tau <= 0):
        raise ValueError(
            "tau values must be positive.\
If you want certain tau values to be zero, then\
run the estimate without those domains."
        )
    if np.sum(tau) < 1:
        raise ValueError("Projection is infeasible because sum of tau values is < 1.")


def linear(estimate, tau):
    """
    TODO
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
            print(
                f"Warning: numerical issues likely caused violation of\
tau bounds: {1-np.sum(projected_estimate)} > {tau_sort[find_index]}"
            )
        projected_estimate[indices_of_largest_to_smallest[find_index]] = 1 - np.sum(
            estimate
        )
    return projected_estimate


def l2(estimate, tau, atol=1e-12):
    """
    TODO
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
