import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Float, Int, Array
from pt_jax._types import RandomKey


def _create_indices(m):
    N = m.shape[0] + 1
    base_indices = jnp.arange(1, N - 1)  # Length N-2
    ind_middle = base_indices + m[1:] - m[:-1]

    ind = jnp.concatenate((jnp.array([m[0]]), ind_middle, jnp.array([N - 1 - m[-1]])))

    return ind


def controlled_swapping(
    x: Float[Array, "n_chains *dims"],
    m: Int[Array, " n_chains-1"],
) -> Float[Array, "n_chains *dims"]:
    """Swaps the entries of `x`, as described by binary mask `m`.

    Args:
      x: array of shape (n_chains, dim)
      m: binary mask of shape (n_chains - 1,)
         controlling which chains should be swapped.
         We have `m[i] = 1` if `x[i]` and `x[i+1]` should be swapped.

    Note:
      Consecutive values 1 in `m` are not allowed.
      Namely, it cannot hold that `m[i] = m[i+1] = 1`.
    """
    indices = _create_indices(m)
    return x[indices, ...]


def generate_deo_extended_kernel(
    log_prob,
    log_ref,
    annealing_schedule,
):
    def log_p(y, beta):
        return beta * log_prob(y) + (1.0 - beta) * log_ref(y)

    log_p_vmap = jax.vmap(log_p, in_axes=(0, 0))

    def extended_kernel(
        key,
        state,
        timestep: int,
    ) -> tuple:
        """Extended deterministic even-odd swap kernel, which
        for even timesteps makes even swaps (2i <-> 2i+1)
        and for odd timesteps makes odd swaps (2i-1 <-> 2i)

        Args:
          key: random key
          state: state
          timestep: timestep number, used to decide whether to make even or odd move

        Returns:
          new_state, the same shape as `state`
          rejection_rates, shape (n_chains-1,)
        """
        n_chains = state.shape[0]

        idx1 = jnp.arange(n_chains - 1)
        idx2 = idx1 + 1

        xs1 = state[idx1]
        xs2 = state[idx2]

        betas1 = annealing_schedule[idx1]
        betas2 = annealing_schedule[idx2]

        log_numerator = log_p_vmap(xs1, betas2) + log_p_vmap(xs2, betas1)
        log_denominator = log_p_vmap(xs1, betas1) + log_p_vmap(xs2, betas2)
        log_accept = log_numerator - log_denominator
        accept_prob = jnp.minimum(jnp.exp(log_accept), 1.0)
        rejection_rates = 1.0 - accept_prob

        # Where the swaps would be accepted through M-H
        accept_mask = jrandom.bernoulli(key, p=accept_prob)
        # Where the swaps can be accepted due to even-odd moves
        even_odd_mask = jnp.mod(idx1, 2) == jnp.mod(timestep, 2)
        total_mask = accept_mask & even_odd_mask

        # Now the tricky part: we need to execute the swaps
        new_state = controlled_swapping(state, total_mask)
        return new_state, rejection_rates

    return extended_kernel


def deo_sampling_loop(
    key: RandomKey,
    x0,
    kernel_local,
    kernel_deo,
    n_samples: int,
    warmup: int,
) -> tuple:
    """The sampling loop for DEO parallel tempering.

    Returns:
      samples
      rejection_rates
    """

    def f(x, timestep: int):
        subkey = jrandom.fold_in(key, timestep)

        key_local, key_deo = jrandom.split(subkey)

        # Apply local exploration kernel
        x = kernel_local(key_local, x)

        # Apply the DEO swap
        x, rejection_rates = kernel_deo(
            key_deo,
            x,
            timestep,
        )

        return x, (x, rejection_rates)

    # Run warmup
    x0, _ = jax.lax.scan(f, x0, jnp.arange(warmup))

    # Collect samples
    _, (samples, rejection_rates) = jax.lax.scan(f, x0, jnp.arange(n_samples))

    return samples, rejection_rates
