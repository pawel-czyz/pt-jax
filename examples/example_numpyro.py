"""Example of using the integration with NumPyro models."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import blackjax

import pt_jax
import pt_jax.numpyro as pt_jax_numpyro


def generate_model(data=None, posterior=True):
    def model():
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        sigma = numpyro.sample("sigma", dist.Exponential(1 / 5.0))

        # u = numpyro.deterministic("u", mu / sigma)

        if posterior:
            with numpyro.plate("data_plate", len(data)):
                numpyro.sample("data", dist.Normal(mu, sigma), obs=data)

    return model


def mala_kernel_generator(log_p, step_size):
    """Wrapper around the MALA kernel from BlackJAX."""
    mala = blackjax.mala(log_p, step_size)

    def kernel(key, position):
        state = mala.init(position)
        new_state, info = mala.step(key, state)
        return new_state.position

    return kernel


def sampling_fn(key, wrapped, betas, x0, n_samples: int = 10_000, warmup: int = 1000):
    # We know how to sample from the reference distribution
    n_chains = len(betas)
    step_sizes = jnp.linspace(0.5, 0.01, n_chains)

    log_target = wrapped.log_posterior_z
    log_ref = wrapped.log_prior_z

    K_ind = pt_jax.kernels.generate_sample_from_prior_kernel(
        log_prob=log_target,
        log_ref=log_ref,
        kernel_ref=wrapped.sample_prior_z,
        truncated_annealing_schedule=betas[1:],
        kernel_generator=mala_kernel_generator,
        truncated_params=step_sizes[1:],
    )
    K_deo = pt_jax.deo.generate_deo_extended_kernel(
        log_prob=log_target,
        log_ref=log_ref,
        annealing_schedule=betas,
    )

    key, subkey = jax.random.split(key)
    samples, _ = pt_jax.deo.deo_sampling_loop(
        key=subkey,
        x0=jnp.zeros([n_chains] + list(x0.shape)),
        kernel_local=K_ind,
        kernel_deo=K_deo,
        n_samples=n_samples,
        warmup=warmup,
    )
    return samples


def main():
    key = jax.random.PRNGKey(101)
    key, subkey = jax.random.split(key)
    data = 3.0 + 2.0 * jax.random.normal(subkey, shape=(80,))

    posterior_model = generate_model(data, posterior=True)
    prior_model = generate_model(posterior=False)

    wrapped = pt_jax_numpyro.wrap_models(prior_model, posterior_model)

    key, subkey = jax.random.split(key)
    init_z = wrapped.sample_prior_z(subkey)

    key, subkey = jax.random.split(key)
    n_chains = 20
    betas = pt_jax.annealing.annealing_exponential(n_chains)
    # Sample in the Z space (unconstrained vectors)
    samples_z = sampling_fn(subkey, wrapped, betas, init_z)
    # Move back to the X space (constrained dictionaries)

    samples_x_prior = jax.vmap(wrapped.z_to_x)(samples_z[:, 0, ...])
    samples_x_posterior = jax.vmap(wrapped.z_to_x)(samples_z[:, -1, ...])

    for k in samples_x_prior.keys():
        print(f"--- Parameter {k} ---")
        mean = samples_x_prior[k].mean()
        std = samples_x_prior[k].std()
        print(f"  Prior:     {float(mean):.2f} +- {float(std):.2f}")

        mean = samples_x_posterior[k].mean()
        std = samples_x_posterior[k].std()
        print(f"  Posterior: {float(mean):.2f} +- {float(std):.2f}\n")


if __name__ == "__main__":
    main()
