"""Utilities for making NumPyro models
suitable for parallel tempering sampling.

The most important constraints are:

  - No `numpyro.deterministic` sites are allowed.
  - Only sample the variables from prior in the prior model.
"""
import jax
from numpyro.infer import Predictive
from numpyro.infer.util import get_transforms, transform_fn, log_density

from typing import Any, NamedTuple


def _get_prior_sampling_fn(prior_model):
    predictive = Predictive(prior_model, num_samples=1)

    def fn(rng_key, *args, **kwargs):
        prior_samples = predictive(rng_key)
        return jax.tree.map(lambda x: x[0], prior_samples)

    return fn


def get_logprob(model):
    def fn(params):
        lp, _ = log_density(model, (), {}, params)
        return lp

    return fn


class Transforms(NamedTuple):
    x_to_u: Any
    u_to_x: Any

    z_to_u: Any
    u_to_z: Any

    x_to_z: Any
    z_to_x: Any

    transforms: Any


def wrap_transforms(transforms: dict, example_params: dict) -> Transforms:
    """Transforms: transform u -> x."""

    def u_to_x(u):
        return transform_fn(transforms, u, invert=False)

    def x_to_u(x):
        return transform_fn(transforms, x, invert=True)

    _, z_to_u = jax.flatten_util.ravel_pytree(x_to_u(example_params))

    def u_to_z(u):
        z, _ = jax.flatten_util.ravel_pytree(u)
        return z

    def x_to_z(x):
        u = x_to_u(x)
        return u_to_z(u)

    def z_to_x(z):
        u = z_to_u(z)
        return u_to_x(u)

    return Transforms(
        x_to_u=x_to_u,
        u_to_x=u_to_x,
        u_to_z=u_to_z,
        z_to_u=z_to_u,
        z_to_x=z_to_x,
        x_to_z=x_to_z,
        transforms=transforms,
    )


def get_model_transforms(model):
    example_params = _get_prior_sampling_fn(model)(jax.random.PRNGKey(0))
    unconstrain_transforms = get_transforms(
        model, model_args=(), model_kwargs={}, params=example_params
    )
    return wrap_transforms(unconstrain_transforms, example_params=example_params)


class LogDensities(NamedTuple):
    logp_x: Any
    logp_u: Any
    logp_z: Any


def reparameterize_logdensity(logp_x_fn, transforms: Transforms) -> LogDensities:
    def logp_u(u):
        # Evaluate the x coordinate and the logprob there
        x = transforms.u_to_x(u)
        logpx = logp_x_fn(x)

        # Correct it by the log-Jacobian of the transformation
        correction = sum(
            [
                transform.log_abs_det_jacobian(u[key], x[key]).sum()
                for key, transform in transforms.transforms.items()
            ]
        )
        return logpx + correction

    def logp_z(z):
        # This transformation is volume-preserving,
        # so no Jacobian correction is required
        return logp_u(transforms.z_to_u(z))

    return LogDensities(
        logp_x=logp_x_fn,
        logp_u=logp_u,
        logp_z=logp_z,
    )


class ParallelTemperingSetting(NamedTuple):
    log_prior_z: Any
    log_posterior_z: Any
    sample_prior_z: Any

    z_to_x: Any


def wrap_models(
    prior,
    posterior,
) -> ParallelTemperingSetting:
    sample_x_fn = _get_prior_sampling_fn(prior)
    transforms = get_model_transforms(prior)

    def sample_z_fn(key, *args, **kwargs):
        x = sample_x_fn(key)
        return transforms.x_to_z(x)

    log_prior = reparameterize_logdensity(
        get_logprob(prior),
        transforms=transforms,
    )
    log_posterior = reparameterize_logdensity(
        get_logprob(posterior),
        transforms=transforms,
    )

    return ParallelTemperingSetting(
        log_posterior_z=log_posterior.logp_z,
        log_prior_z=log_prior.logp_z,
        sample_prior_z=sample_z_fn,
        z_to_x=transforms.z_to_x,
    )
