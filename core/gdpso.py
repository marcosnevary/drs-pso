from typing import NamedTuple

import numpy as np


class SwarmState(NamedTuple):
    positions: np.ndarray
    velocities: np.ndarray
    p_best_pos: np.ndarray
    p_best_fit: np.ndarray
    g_best_pos: np.ndarray
    g_best_fit: np.ndarray
    rng: np.random.Generator
    history: np.ndarray


def numerical_gradient(objective_fn: callable, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon
        grad[i] = (objective_fn(x_plus) - objective_fn(x_minus)) / (2 * epsilon)
    return grad


def gradient_descent(
    objective_fn: callable,
    initial_pos: np.ndarray,
    bounds: tuple,
    eta: float,
    steps: int,
) -> tuple:
    lower, upper = bounds
    current_pos = initial_pos.copy()

    for _ in range(steps):
        grad = numerical_gradient(objective_fn, current_pos)
        current_pos = current_pos - eta * grad
        current_pos = np.clip(current_pos, lower, upper)

    final_fit = objective_fn(current_pos)
    return current_pos, final_fit


def gdpso(
    objective_fn: callable,
    bounds: tuple,
    num_dims: int,
    num_particles: int,
    max_iters: int,
    c1: float,
    c2: float,
    w: float,
    seed: int,
    eta: float,
    steps: int,
    gd_interval: int,
    **_: any,
) -> tuple:
    lower, upper = bounds
    rng = np.random.default_rng(seed)

    init_positions = rng.uniform(lower, upper, (num_particles, num_dims))
    init_velocities = np.zeros((num_particles, num_dims))
    init_fitness = np.array([objective_fn(position) for position in init_positions])

    best_idx = np.argmin(init_fitness)
    g_best_pos = init_positions[best_idx]
    g_best_fit = init_fitness[best_idx]

    history = np.zeros(max_iters)
    history[0] = g_best_fit

    swarm_state = SwarmState(
        positions=init_positions,
        velocities=init_velocities,
        p_best_pos=init_positions,
        p_best_fit=init_fitness,
        g_best_pos=g_best_pos,
        g_best_fit=g_best_fit,
        rng=rng,
        history=history,
    )

    for i in range(max_iters):
        r1 = swarm_state.rng.random((num_particles, num_dims))
        r2 = swarm_state.rng.random((num_particles, num_dims))

        inertia = w * swarm_state.velocities
        cognitive = c1 * r1 * (swarm_state.p_best_pos - swarm_state.positions)
        social = c2 * r2 * (swarm_state.g_best_pos - swarm_state.positions)

        new_velocities = inertia + cognitive + social
        new_positions = swarm_state.positions + new_velocities
        new_positions = np.clip(new_positions, lower, upper)

        new_fitness = np.array([objective_fn(pos) for pos in new_positions])

        improved = new_fitness < swarm_state.p_best_fit
        mask = improved[:, None]
        new_p_best_pos = np.where(mask, new_positions, swarm_state.p_best_pos)
        new_p_best_fit = np.where(improved, new_fitness, swarm_state.p_best_fit)

        current_g_best_idx = np.argmin(new_p_best_fit)
        current_g_best_fit = new_p_best_fit[current_g_best_idx]
        global_improved = current_g_best_fit < swarm_state.g_best_fit

        candidate_g_pos = np.where(
            global_improved,
            new_p_best_pos[current_g_best_idx],
            swarm_state.g_best_pos,
        )
        candidate_g_fit = np.where(
            global_improved,
            current_g_best_fit,
            swarm_state.g_best_fit,
        )

        if i % gd_interval == 0:
            gradient_g_pos, gradient_g_fit = gradient_descent(
                objective_fn,
                candidate_g_pos,
                bounds,
                eta,
                steps,
            )

            gd_improved = gradient_g_fit < candidate_g_fit
            final_g_pos = gradient_g_pos if gd_improved else candidate_g_pos
            final_g_fit = gradient_g_fit if gd_improved else candidate_g_fit

            if gd_improved and (final_g_fit < swarm_state.g_best_fit):
                new_p_best_pos[current_g_best_idx] = final_g_pos
                new_p_best_fit[current_g_best_idx] = final_g_fit
        else:
            final_g_pos = candidate_g_pos
            final_g_fit = candidate_g_fit

        new_history = swarm_state.history
        new_history[i] = final_g_fit

        swarm_state = SwarmState(
            positions=new_positions,
            velocities=new_velocities,
            p_best_pos=new_p_best_pos,
            p_best_fit=new_p_best_fit,
            g_best_pos=final_g_pos,
            g_best_fit=final_g_fit,
            rng=swarm_state.rng,
            history=new_history,
        )

    return swarm_state.g_best_pos, swarm_state.g_best_fit, swarm_state.history
