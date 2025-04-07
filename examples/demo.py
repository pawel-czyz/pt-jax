"""Demo: parallel tempering for sampling from
a two-dimensional distribution employing the MALA kernel
from BlackJAX."""
try:
    import matplotlib.pyplot as plt
    import numpyro.distributions as dist
    import blackjax
except ModuleNotFoundError:
    print(
        "Use `pip install matplotlib numpyro` to install additional "
        "packages for this example."
    )

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pt_jax


def log_ref(x):
    return dist.Normal(jnp.zeros(2), 2.0).to_event().log_prob(x)


def log_target(x):
    c0 = jnp.asarray([-2.0, -2.0])
    c1 = jnp.asarray([0.0, 0.0])
    c2 = jnp.asarray([2.0, 2.0])

    t = 50.0

    l0 = -t * (jnp.sum(jnp.square(x - c0)) - 1.0**2) ** 2
    l1 = -t * (jnp.sum(jnp.square(x - c1)) - 1.0**2) ** 2
    l2 = -t * (jnp.sum(jnp.square(x - c2)) - 1.0**2) ** 2

    return jax.nn.logsumexp(jnp.array([l0, l1, l2]))


def mala_kernel_generator(log_p, step_size):
    """Wrapper around the MALA kernel from BlackJAX."""
    mala = blackjax.mala(log_p, step_size)

    def kernel(key, position):
        state = mala.init(position)
        new_state, info = mala.step(key, state)
        return new_state.position

    return kernel


def sampling_fn(key, betas, x0, n_samples: int = 5000, warmup: int = 1000):
    # We know how to sample from the reference distribution
    n_chains = len(betas)
    step_sizes = jnp.linspace(0.5, 0.01, n_chains)

    K_ind = pt_jax.kernels.generate_independent_annealed_kernel(
        log_prob=log_target,
        log_ref=log_ref,
        annealing_schedule=betas,
        kernel_generator=mala_kernel_generator,
        params=step_sizes,
    )
    K_deo = pt_jax.deo.generate_deo_extended_kernel(
        log_prob=log_target,
        log_ref=log_ref,
        annealing_schedule=betas,
    )

    key, subkey = jrandom.split(key)
    # Note that we could use the rejections information
    # (currently ignored as `_`) to optimize the annealing schedule
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
    key = jax.random.PRNGKey(2025 - 4 - 7)
    x0 = jnp.zeros(2)

    plot_dims = (4, 5)
    n_chains = plot_dims[0] * plot_dims[1]

    betas = pt_jax.annealing.annealing_exponential(n_chains)

    samples = sampling_fn(key, betas, x0)
    fig, axs = plt.subplots(*plot_dims, sharex=True, sharey=True)

    for i, ax in enumerate(axs.ravel()):
        smp = samples[:, i, :]  # (n_samples, n_chains, n_dims)
        thinning = 10
        ax.scatter(smp[::thinning, 0], smp[::thinning, 1], s=2, c="k", rasterized=True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"$\\beta={betas[i]:.3f}$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    fig.tight_layout()
    fig.savefig("plot.pdf")


if __name__ == "__main__":
    main()
