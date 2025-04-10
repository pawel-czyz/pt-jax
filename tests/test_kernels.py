import jax
import jax.numpy as jnp
import numpy.testing as npt

import pytest
import pt_jax.kernels as K


@pytest.mark.parametrize("n_chains", [3, 5])
@pytest.mark.parametrize("shape", [(3,)])
def test_smoke_generate_sample_from_prior_kernel(n_chains: int, shape: tuple):
    def log_p(x):
        return -1.0

    def kernel_generator(log_p, param):
        def k(key, x):
            return x * param

        return k

    params = jnp.linspace(-1, 5, n_chains)
    temperature = jnp.linspace(0, 1, n_chains)

    kernel_ref = kernel_generator(log_p, 1.0)

    kernel = K.generate_sample_from_prior_kernel(
        log_prob=log_p,
        log_ref=log_p,
        kernel_ref=kernel_ref,
        kernel_generator=kernel_generator,
        truncated_annealing_schedule=temperature[1:],
        truncated_params=params[1:],
    )

    x0 = jnp.ones([n_chains] + list(shape))
    key = jax.random.PRNGKey(42)
    x1 = kernel(key, x0)

    assert x1.shape == x0.shape
    # Only works for deterministic kernels
    npt.assert_allclose(x1[0, ...], kernel_ref(key, x0[0, ...]))
