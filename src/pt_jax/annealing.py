import jax.numpy as jnp


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
