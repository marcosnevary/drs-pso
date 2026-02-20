from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import jit, lax, random, vmap


class JaxSwarmState(NamedTuple):
    positions: jnp.ndarray
    velocities: jnp.ndarray
    p_best_pos: jnp.ndarray
    p_best_fit: jnp.ndarray
    g_best_pos: jnp.ndarray
    g_best_fit: jnp.ndarray
    rng: random.PRNGKey


@partial(
    jit,
    static_argnames=(
        "objective_fn",
        "num_dims",
        "num_particles",
        "max_iters",
    ),
)
def pso_jax(
    objective_fn: Callable,
    bounds: tuple,
    num_dims: int,
    num_particles: int,
    max_iters: int,
    c1: float,
    c2: float,
    w: float,
    seed: random.PRNGKey,
    **_: any,
) -> tuple:
    key = seed
    lower, upper = jnp.array(bounds[0]), jnp.array(bounds[1])
    k_pos, k_vel, k_state = random.split(key, 3)

    search_range = upper - lower
    velocity_scale = 0.1
    limit = search_range * velocity_scale

    init_positions = random.uniform(k_pos, (num_particles, num_dims), minval=lower, maxval=upper)
    init_velocities = random.uniform(k_vel, (num_particles, num_dims), minval=-limit, maxval=limit)
    init_fitness = vmap(objective_fn)(init_positions)

    best_idx = jnp.argmin(init_fitness)
    g_best_pos = init_positions[best_idx]
    g_best_fit = init_fitness[best_idx]

    initial_state = JaxSwarmState(
        positions=init_positions,
        velocities=init_velocities,
        p_best_pos=init_positions,
        p_best_fit=init_fitness,
        g_best_pos=g_best_pos,
        g_best_fit=g_best_fit,
        rng=k_state,
    )

    def update_step(swarm_state: JaxSwarmState, _: any) -> tuple:
        k1, k2, k_next = random.split(swarm_state.rng, 3)
        r1 = random.uniform(k1, (num_particles, num_dims))
        r2 = random.uniform(k2, (num_particles, num_dims))

        inertia = w * swarm_state.velocities
        cognitive = c1 * r1 * (swarm_state.p_best_pos - swarm_state.positions)
        social = c2 * r2 * (swarm_state.g_best_pos - swarm_state.positions)

        new_velocities = inertia + cognitive + social
        new_positions = swarm_state.positions + new_velocities
        new_positions = jnp.clip(new_positions, lower, upper)

        new_fitness = vmap(objective_fn)(new_positions)

        improved = new_fitness < swarm_state.p_best_fit

        new_p_best_pos = jnp.where(improved[:, None], new_positions, swarm_state.p_best_pos)
        new_p_best_fit = jnp.where(improved, new_fitness, swarm_state.p_best_fit)

        current_g_best_idx = jnp.argmin(new_p_best_fit)
        new_g_best_pos = new_p_best_pos[current_g_best_idx]
        new_g_best_fit = new_p_best_fit[current_g_best_idx]

        global_improved = new_g_best_fit < swarm_state.g_best_fit

        final_g_best_pos = jnp.where(global_improved, new_g_best_pos, swarm_state.g_best_pos)
        final_g_best_fit = jnp.where(global_improved, new_g_best_fit, swarm_state.g_best_fit)

        next_state = JaxSwarmState(
            positions=new_positions,
            velocities=new_velocities,
            p_best_pos=new_p_best_pos,
            p_best_fit=new_p_best_fit,
            g_best_pos=final_g_best_pos,
            g_best_fit=final_g_best_fit,
            rng=k_next,
        )

        return next_state, final_g_best_fit

    final_state, history = lax.scan(update_step, initial_state, jnp.arange(max_iters))
    full_history = jnp.concatenate([jnp.array([initial_state.g_best_fit]), history])

    return final_state.g_best_pos, final_state.g_best_fit, full_history
