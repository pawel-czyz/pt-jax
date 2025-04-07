import jax
import jax.numpy as jnp
import jax.random as jrandom


def generate_independent_annealed_kernel(
    log_prob,
    log_ref,
    annealing_schedule,
    kernel_generator,
    params,
) -> tuple:
    """Generates the kernels via the kernel generator given appropriate parameters.

    Args:
      log_prob: log_prob of the target distribution
      log_ref: log_prob of the easy-to-sample reference distribution
      annealing_schedule: annealing schedule such that
        `annealing_schedule[0] = 0.0` and `annealing_schedule[-1] = 1`
      kernel_generator: `kernel_generator(log_p, param)`
        returns a transition kernel of signature `kernel(key, state) -> new_state`
      params: parameters for the transition kernels.
        Note that `len(annealing_schedule) = len(params)`
    """
    if len(annealing_schedule) != len(params):
        raise ValueError(
            "Parameters have to be of the same length as the annealing schedule"
        )
    n_chains = len(annealing_schedule)

    def transition_kernel(key, state, beta, param):
        def log_p(y):
            return beta * log_prob(y) + (1.0 - beta) * log_ref(y)

        return kernel_generator(log_p, param)(key, state)

    def kernel(key, state_joint):
        key_vec = jrandom.split(key, n_chains)
        return jax.vmap(transition_kernel, in_axes=(0, 0, 0, 0))(
            key_vec, state_joint, annealing_schedule, params
        )

    return kernel


def generate_swap_chains_decision_kernel(
    log_prob,
    log_ref,
    annealing_schedule,
):
    def log_p(y, beta):
        return beta * log_prob(y) + (1.0 - beta) * log_ref(y)

    def swap_decision(key, state, i: int, j: int) -> bool:
        beta1, beta2 = annealing_schedule[i], annealing_schedule[j]
        x1, x2 = state[i], state[j]
        log_numerator = log_p(x1, beta2) + log_p(x2, beta1)
        log_denominator = log_p(x1, beta1) + log_p(x2, beta2)
        log_r = log_numerator - log_denominator

        r = jnp.exp(log_r)
        return jrandom.uniform(key) < r

    return swap_decision


def generate_full_sweep_swap_kernel(
    log_prob,
    log_ref,
    annealing_schedule,
):
    """Applies a full sweep, attempting to swap chains 0 <-> 1,
    then 1 <-> 2 etc. one-after-another."""
    n_chains = len(annealing_schedule)

    if n_chains < 2:
        raise ValueError("At least two chains are needed.")

    swap_decision_fn = generate_swap_chains_decision_kernel(
        log_prob=log_prob,
        log_ref=log_ref,
        annealing_schedule=annealing_schedule,
    )

    def kernel(key, state):
        def f(state, i: int):
            subkey = jrandom.fold_in(key, i)
            decision = swap_decision_fn(subkey, state=state, i=i, j=i + 1)

            # Candidate state: we swap values at i and i+1 positions
            swapped_state = state.at[i].set(state[i + 1])
            swapped_state = swapped_state.at[i + 1].set(state[i])

            new_state = jax.lax.select(decision, swapped_state, state)
            return new_state, None

        final_state, _ = jax.lax.scan(f, state, jnp.arange(n_chains - 1))
        return final_state

    return kernel


def compose_kernels(kernels: list):
    """Composes kernels, applying them in order."""

    def kernel(key, state):
        for ker in kernels:
            key, subkey = jrandom.split(key)
            state = ker(subkey, state)

        return state

    return kernel
