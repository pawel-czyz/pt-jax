"""Spike and slab prior."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax.random as random
from numpyro.infer import Predictive, MCMC, NUTS
import matplotlib.pyplot as plt

from numpyro.distributions import Distribution, constraints, Normal


class ContinuousSpikeSlab(Distribution):
    """
    A continuous "spike-and-slab" prior:
      p(beta | w) = (1 - w)*Normal(0, scale_spike) + w*Normal(0, scale_slab),
    treated as a single distribution in beta (no discrete mixture index).

    Parameters
    ----------
    w : float or array
        Mixing weight in [0, 1], can be broadcast if needed.
    loc_spike : float or array
        Location (mean) of "spike" component (default 0).
    scale_spike : float or array
        Scale (std dev) of "spike" component (default 0.1).
    loc_slab : float or array
        Location (mean) of "slab" component (default 0).
    scale_slab : float or array
        Scale (std dev) of "slab" component (default 3.0).
    """

    arg_constraints = {
        "w": constraints.unit_interval,
        "loc_spike": constraints.real,
        "scale_spike": constraints.positive,
        "loc_slab": constraints.real,
        "scale_slab": constraints.positive,
    }
    support = constraints.real
    # Not reparameterizable in the usual sense
    has_rsample = False

    def __init__(
        self,
        w,
        loc_spike=0.0,
        scale_spike=0.1,
        loc_slab=0.0,
        scale_slab=3.0,
        validate_args=None,
    ):
        self.w = w
        self.loc_spike = loc_spike
        self.scale_spike = scale_spike
        self.loc_slab = loc_slab
        self.scale_slab = scale_slab

        # Construct sub-distributions for spike and slab
        self.spike_dist = Normal(loc_spike, scale_spike)
        self.slab_dist = Normal(loc_slab, scale_slab)

        # Figure out the broadcasted batch shape of all parameters
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(w),
            jnp.shape(loc_spike),
            jnp.shape(scale_spike),
            jnp.shape(loc_slab),
            jnp.shape(scale_slab),
        )
        super().__init__(
            batch_shape=batch_shape, event_shape=(), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        """
        Draw a sample from the mixture. This is not reparameterized
        (it still internally draws a Bernoulli to choose spike vs slab),
        but is fine for *forward sampling* or generating synthetic data.
        """
        # We flatten sample_shape + batch_shape into a single size to draw Bernoulli
        numel = jnp.prod(jnp.array(sample_shape + self.batch_shape))

        # Split key for mixture indicator vs. the actual normal draws
        key_indicator, key_spike, key_slab = jax.random.split(key, 3)

        # Mixture indicators in {0,1}, shape = (numel,)
        # p= self.w, but we must broadcast self.w to match
        mixture_indicator = jax.random.bernoulli(
            key_indicator, p=jnp.broadcast_to(self.w, (numel,))
        )

        # Draw from spike and slab
        spike_samples = self.spike_dist.sample(key_spike, sample_shape=(numel,))
        slab_samples = self.slab_dist.sample(key_slab, sample_shape=(numel,))

        # Pick slab where mixture_indicator=1, else spike
        samples = jnp.where(mixture_indicator, slab_samples, spike_samples)

        # Reshape back to sample_shape + batch_shape
        out_shape = sample_shape + self.batch_shape
        return jnp.reshape(samples, out_shape)

    def log_prob(self, value):
        """
        The key: a single continuous log_prob:
           log( (1-w)*p_spike(x) + w*p_slab(x) )
        which is fully differentiable wrt 'value'.
        """
        log_spike = self.spike_dist.log_prob(value)
        log_slab = self.slab_dist.log_prob(value)

        # Broadcast w to match the shape of value if needed
        w_ = jnp.broadcast_to(self.w, jnp.shape(log_spike))

        # Use log-sum-exp for numerical stability
        log_mix_spike = jnp.log1p(-w_) + log_spike  # log((1-w)*p_spike)
        log_mix_slab = jnp.log(w_) + log_slab  # log(w*p_slab)

        return jnp.logaddexp(log_mix_spike, log_mix_slab)


def spike_and_slab_model(X, y=None, sigma=None, w=None):
    n_obs, n_vars = X.shape

    # Place a prior on w (the slab mixing weight)
    w = numpyro.sample("w", dist.Beta(2, 8), obs=w)

    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0), obs=sigma)

    beta = numpyro.sample(
        "beta",
        ContinuousSpikeSlab(
            w=w, loc_spike=0.0, scale_spike=0.02, loc_slab=0.0, scale_slab=3.0
        ).expand([n_vars]),
    )

    mu = jnp.einsum("ng,g->n", X, beta)
    with numpyro.plate("data", n_obs):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


def main():
    rng_key = random.PRNGKey(45)
    rng_key, key2 = random.split(rng_key)
    # (n_points, n_features)
    X = random.normal(key2, shape=(200, 50))

    # Generate one set of synthetic data
    predictive = Predictive(
        spike_and_slab_model,
        num_samples=1,
    )
    synthetic_data = predictive(rng_key, X, None, 1.0)

    # Extract generated components:
    y_generated = synthetic_data["y"][0]
    beta_generated = synthetic_data["beta"][0]
    synthetic_data["sigma"][0]
    synthetic_data["w"][0]

    # Initialize the NUTS kernel with the spike-and-slab model
    nuts_kernel = NUTS(spike_and_slab_model)

    # Set up the MCMC sampler with desired warmup and sample sizes
    mcmc = MCMC(nuts_kernel, num_warmup=3_000, num_samples=5000, num_chains=4)

    # Run MCMC. Pass y=None to generate synthetic data or provide your observed data.
    mcmc.run(rng_key, X=X, y=y_generated)  # , w=0.99)

    # Optionally, print a summary of the inferred posterior distributions.
    mcmc.print_summary()

    fig, axs = plt.subplots(1, 2)
    true_beta = beta_generated

    ax = axs[0]
    x_ax = jnp.arange(true_beta.shape[0])

    order = jnp.argsort(true_beta)

    ax.set_xlabel("Index $i$")
    ax.set_ylabel("Coefficient $\\beta_i$")

    ax.plot(x_ax, true_beta[order], c="maroon", linewidth=2.0)

    for sample in mcmc.get_samples()["beta"][::100, ...]:
        ax.plot(x_ax, sample[order], c="blue", alpha=0.01)

    ax = axs[1]

    t = jnp.linspace(jnp.min(true_beta), jnp.max(true_beta), 21)

    ax.plot(t, t, linestyle="--", linewidth=1, c="k")
    ax.scatter(true_beta[order], true_beta[order])
    ax.set_xlabel("$\\beta_i$")
    ax.set_ylabel("$\\beta_i$")

    ax.set_xlim(1.05 * t.min(), 1.05 * t.max())
    ax.set_ylim(1.05 * t.min(), 1.05 * t.max())

    for sample in mcmc.get_samples()["beta"][::100, ...]:
        ax.plot(true_beta[order], sample[order], c="blue", alpha=0.01)

    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("spike-and-slab.pdf")


if __name__ == "__main__":
    main()
