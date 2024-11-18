"""
Utility functions for alignment of spherical signals.

This module provides functions for aligning spherical signals through quaternion-based
rotations. It includes methods for sampling random quaternions, computing distances
between signals on a sphere, and optimizing rotational alignment using gradient descent.
"""
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import optax


@jax.jit
def sample_uniform_quaternion(key):
    """Sample a uniform random quaternion using Arvo's method.
    
    Implements the algorithm from "Uniform Random Rotations" by James Arvo
    to generate uniformly distributed random quaternions representing 3D rotations.
    
    Args:
        key: JAX PRNG key for random number generation
    
    Returns:
        jnp.ndarray: A normalized quaternion as a 4D array [w, x, y, z] representing
            a random rotation
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
    """Sample multiple uniform random quaternions in parallel.
    
    Args:
        key: JAX PRNG key for random number generation
        num_samples: Number of quaternions to sample
    
    Returns:
        jnp.ndarray: Array of shape (num_samples, 4) containing the sampled quaternions,
            where each quaternion is represented as [w, x, y, z]
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
    """Normalize a quaternion to unit length.
    
    Args:
        q: Input quaternion as a 4D array [w, x, y, z]
    
    Returns:
        jnp.ndarray: Normalized quaternion with unit length
    """
    return q / jnp.linalg.norm(q)


@jax.jit
def signal_to_s2grid_vectors(signal):
    """Convert a spherical signal to grid vectors.
    
    Args:
        signal: Input spherical signal
        
    Returns:
        jnp.ndarray: Array of shape (N, 3) containing the grid vectors
    """
    signal_grid = e3nn.to_s2grid(signal, 30, 29, quadrature="soft")
    signal_vectors = signal_grid.grid_vectors * signal_grid.grid_values[..., None]
    signal_vectors = signal_vectors.reshape((-1, 3))
    return signal_vectors


@jax.jit
def spherical_distance(signal1, signal2):
    """Calculate the distance between two spherical signals.
    
    Computes the mean minimum squared Euclidean distance between the grid points
    of two spherical signals.
    
    Args:
        signal1: First spherical signal
        signal2: Second spherical signal
        
    Returns:
        float: Mean minimum squared distance between the signals
    """
    signal1_vectors = signal_to_s2grid_vectors(signal1)
    signal2_vectors = signal_to_s2grid_vectors(signal2)
    
    squared_distances = jnp.linalg.norm(
        signal1_vectors[:, None, :] - signal2_vectors[None, :, :], 
        axis=-1
    ) ** 2
    return jnp.mean(jnp.min(squared_distances, axis=-1))


@jax.jit
def loss_fn(quaternion, signal1, signal2):
    """Compute the alignment loss between two signals under a rotation.
    
    Args:
        quaternion: Rotation quaternion to apply to signal1
        signal1: First spherical signal to be rotated
        signal2: Second spherical signal (target)
        
    Returns:
        float: Distance between the rotated signal1 and signal2
    """
    rotated_signal1 = signal1.transform_by_quaternion(quaternion)
    return spherical_distance(rotated_signal1, signal2)


def find_best_random_quaternion(key, signal1, signal2, num_samples=100):
    """Find the best aligning quaternion from a random sample.
    
    Args:
        key: JAX PRNG key for random sampling
        signal1: First spherical signal to be rotated
        signal2: Second spherical signal (target)
        num_samples: Number of random quaternions to try
        
    Returns:
        jnp.ndarray: Best quaternion found from the random samples
    """
    quaternions = sample_multiple_quaternions(key, num_samples)
    return quaternions[jnp.argmin(jnp.array([loss_fn(q, signal1, signal2) for q in quaternions]))]


def align_signals(
    signal1, 
    signal2, 
    initial_quaternion,
    learning_rate=0.01,
    num_iterations=250,
):
    """Find optimal rotation to align two spherical signals using gradient descent.
    
    Uses Adam optimizer to find the quaternion rotation that best aligns signal1
    with signal2 by minimizing their spherical distance.
    
    Args:
        signal1: First spherical signal to be rotated
        signal2: Second spherical signal (target)
        initial_quaternion: Starting quaternion for optimization
        learning_rate: Learning rate for Adam optimizer
        num_iterations: Number of optimization iterations
        
    Returns:
        jnp.ndarray: Optimized quaternion that best aligns the signals
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
        
    final_loss = loss_fn(current_quaternion, signal1, signal2)

    return current_quaternion, final_loss