import jax.numpy as jnp
import numpy as np
from jax import jit

from .gdpso import gdpso
from .gdpso_jax import gdpso_jax
from .pso import pso
from .pso_jax import pso_jax


def schwefel_np(x: np.ndarray) -> float:
    n = x.shape[0]
    sum_term = np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return 418.9829 * n - sum_term


def rastrigin_np(x: np.ndarray) -> float:
    n = x.shape[0]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def sphere_np(x: np.ndarray) -> float:
    return np.sum(x**2)


def elliptic_np(x: np.ndarray) -> float:
    n = x.shape[0]
    i = np.arange(n)
    coeffs = (1e6) ** (i / (n - 1))
    return np.sum(coeffs * x**2)


@jit
def schwefel_jax(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    sum_term = jnp.sum(x * jnp.sin(jnp.sqrt(jnp.abs(x))))
    return 418.9829 * n - sum_term


@jit
def rastrigin_jax(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    return 10 * n + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))


@jit
def sphere_jax(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x**2)


@jit
def elliptic_jax(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    i = jnp.arange(n)

    coeffs = (1e6) ** (i / (n - 1))
    return jnp.sum(coeffs * x**2)


BENCHMARKS = {
    "Schwefel": {
        "bounds": (-500, 500),
        "NumPy": schwefel_np,
        "JAX": schwefel_jax,
    },
    "Rastrigin": {
        "bounds": (-5.12, 5.12),
        "NumPy": rastrigin_np,
        "JAX": rastrigin_jax,
    },
    "Elliptic": {
        "bounds": (-5.0, 10.0),
        "NumPy": elliptic_np,
        "JAX": elliptic_jax,
    },
    "Sphere": {
        "bounds": (-5.12, 5.12),
        "NumPy": sphere_np,
        "JAX": sphere_jax,
    },
}

ALGORITHMS = {
    "PSO": pso,
    "GDPSO": gdpso,
    "PSO-JAX": pso_jax,
    "GDPSO-JAX": gdpso_jax,
}

DIMS = [10, 30, 50, 100]

HYPERPARAMETERS = {
    "num_dims": None,
    "num_particles": 30,
    "max_iters": 1000,
    "c1": 1.5,
    "c2": 2.5,
    "w": 0.7,
    "seed": None,
    "eta": 0.001,
    "steps": 10,
    "gd_interval": 10,
}

NUM_RUNS = 10
