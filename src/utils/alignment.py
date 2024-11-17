"""
Utility functions for alignment of spherical signals.
"""

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import optax

@jax.jit
def sample_uniform_quaternion(key):
    """
    Sample a uniform random quaternion using the method described in:
    "Uniform Random Rotations" by James Arvo.
    
    Args:
        key: JAX PRNG key
    
    Returns:
        q: normalized quaternion as a 4D array [w, x, y, z]
    """
    # Split the key for multiple random number generations
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Sample three uniform random numbers in [0,1]
    u1 = jax.random.uniform(key1)
    u2 = jax.random.uniform(key2)
    u3 = jax.random.uniform(key3)
    
    # Convert to spherical coordinates
    sqrt1_minus_u1 = jnp.sqrt(1.0 - u1)
    sqrt_u1 = jnp.sqrt(u1)
    theta1 = 2.0 * jnp.pi * u2
    theta2 = 2.0 * jnp.pi * u3
    
    # Compute quaternion components
    w = sqrt1_minus_u1 * jnp.sin(theta1)
    x = sqrt1_minus_u1 * jnp.cos(theta1)
    y = sqrt_u1 * jnp.sin(theta2)
    z = sqrt_u1 * jnp.cos(theta2)
    
    return jnp.array([w, x, y, z])

def sample_multiple_quaternions(key, num_samples):
    """
    Sample multiple uniform random quaternions.
    
    Args:
        key: JAX PRNG key
        num_samples: int, number of quaternions to sample
    
    Returns:
        qs: array of shape (num_samples, 4) containing the sampled quaternions
    """
    # Create multiple keys, one for each sample
    keys = jax.random.split(key, num_samples)
    
    # Use vmap to vectorize the sampling function across all keys
    sample_fn = jax.vmap(sample_uniform_quaternion)
    
    # Sample quaternions in parallel
    quaternions = sample_fn(keys)
    
    return quaternions


@jax.jit
def normalize_quaternion(q):
    """Normalize quaternion to unit length"""
    return q / jnp.linalg.norm(q)


@jax.jit
def spherical_distance(signal1, signal2):
    """Spherical distance calculation"""
    signal1_grid = e3nn.to_s2grid(signal1, 30, 29, quadrature="soft")
    signal2_grid = e3nn.to_s2grid(signal2, 30, 29, quadrature="soft")

    signal1_vectors = signal1_grid.grid_vectors * signal1_grid.grid_values[..., None]
    signal2_vectors = signal2_grid.grid_vectors * signal2_grid.grid_values[..., None]

    signal1_vectors = signal1_vectors.reshape((-1, 3))
    signal2_vectors = signal2_vectors.reshape((-1, 3))
    
    squared_distances = jnp.linalg.norm(
        signal1_vectors[:, None, :] - signal2_vectors[None, :, :], 
        axis=-1
    ) ** 2
    return jnp.mean(jnp.min(squared_distances, axis=-1))


@jax.jit
def loss_fn(quaternion, signal1, signal2):
    """Loss function to minimize"""
    rotated_signal1 = signal1.transform_by_quaternion(quaternion)
    return spherical_distance(rotated_signal1, signal2)


def find_best_random_quaternion(key, signal1, signal2, num_samples=100):
    quaternions = sample_multiple_quaternions(key, num_samples)
    return quaternions[jnp.argmin(jnp.array([loss_fn(q, signal1, signal2) for q in quaternions]))]


def align_signals(
    signal1, 
    signal2, 
    initial_quaternion,
    learning_rate=0.01,
    num_iterations=250,
):
    """
    Find optimal rotation using gradient descent
    
    Parameters:
        signal1: First spherical signal
        signal2: Second spherical signal
        initial_quaternion: Starting quaternion for optimization
        learning_rate: Learning rate for optimizer
        num_iterations: Number of optimization iterations
    Returns:
        Optimized quaternion
    """
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    
    @jax.jit
    def update_step(params, opt_state, signal1, signal2):
        grads = jax.grad(loss_fn)(params, signal1, signal2)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = normalize_quaternion(params)
        return params, opt_state

    # Initialize optimization
    current_quaternion = initial_quaternion
    opt_state = optimizer.init(current_quaternion)
    
    # Run optimization
    for _ in range(num_iterations):
        current_quaternion, opt_state = update_step(
            current_quaternion, opt_state, signal1, signal2
        )
        
    return current_quaternion