import jax.numpy as jnp

from scipy.interpolate import PchipInterpolator
from scipy.optimize import bisect


def annealing_constant(n_chains: int, base: float = 1.0):
    """Constant annealing schedule, should be avoided."""
    return base * jnp.ones(n_chains)


def annealing_linear(n_chains: int):
    """Linear annealing schedule, should be avoided."""
    return jnp.linspace(0.0, 1.0, n_chains)


def annealing_exponential(n_chains: int, base: float = 2.0**0.5):
    """Annealing parameters form a geometric series (apart from beta[0] = 0).

    Args:
      n_chains: number of chains in the schedule
      base: geometric progression base, float larger than 1
    """
    if base <= 1:
        raise ValueError("Base should be larger than 1.")

    if n_chains < 2:
        raise ValueError("At least two chains are required.")
    elif n_chains == 2:
        return jnp.array([0.0, 1.0])
    else:
        x = jnp.append(jnp.power(base, -jnp.arange(n_chains - 1)), 0.0)
        return x[::-1]


def estimate_lambda_values(rejection_rates, offset: float = 1e-3):
    # Make sure that the estimated rejection rates are non-zero
    rejection_rates = jnp.maximum(rejection_rates, offset)
    # We have Lambda(0) = 0 and then estimate the rest by cumulative sums
    extended = jnp.concatenate((jnp.zeros(1), rejection_rates))
    return jnp.cumsum(extended)


def get_lambda_function(
    annealing_schedule,
    lambda_values,
):
    """Approximates the Lambda function from several estimates at the schedule
    by interpolating the values with a monotonic cubic spline (as advised in the paper).
    """
    return PchipInterpolator(annealing_schedule, lambda_values)


def annealing_optimal(
    n_chains: int,
    previous_schedule,
    rejection_rates,
    _offset: float = 1e-3,
):
    """Finds the optimal annealing schedule basing on
    the approximation of the Lambda function."""
    lambda_values = estimate_lambda_values(rejection_rates, offset=_offset)
    lambda_fn = get_lambda_function(
        previous_schedule,
        lambda_values,
    )

    lambda1 = lambda_values[-1]

    new_schedule = [0.0]

    for k in range(1, n_chains - 1):

        def fn(x):
            desired_value = k * lambda1 / (n_chains - 1)
            return lambda_fn(x) - desired_value

        new_point = bisect(
            fn,
            new_schedule[-1],
            1.0,
        )

        if new_point >= 1.0:
            raise ValueError("Encountered value 1.0.")

        new_schedule.append(new_point)

    new_schedule.append(1.0)

    if len(new_schedule) != n_chains:
        raise Exception("This should not happen.")

    return jnp.asarray(new_schedule, dtype=float)
