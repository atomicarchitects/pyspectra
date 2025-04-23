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
def quaternion_rotation_distance(q1, q2):
    """
    Calculate the rotation distance between two quaternions.
    
    Args:
        q1: First quaternion in format [w, x, y, z]
        q2: Second quaternion in format [w, x, y, z]
    
    Returns:
        float: Rotation distance between the two quaternions
    """
    dot = jnp.dot(q1, q2)
    dot = jnp.clip(jnp.abs(dot), -1.0, 1.0)
    return 2 * jnp.arccos(dot)


def evenly_distributed_quaternions(n):
    """
    Generate n evenly distributed quaternions representing unique rotations in SO(3).
    
    This implementation uses a Fibonacci lattice on the 4D hypersphere and
    ensures all quaternions represent unique rotations (accounting for the fact 
    that q and -q represent the same rotation).
    
    Parameters:
    -----------
    n : int
        Number of quaternions to generate
        
    Returns:
    --------
    quaternions : jax.numpy.ndarray
        Array of shape (n, 4) containing n unit quaternions in format [w, x, y, z]
    """
    # Generate points using multiple irrational numbers for better distribution
    indices = jnp.arange(0, n, dtype=float) + 0.5
    
    # Use different irrational numbers to create optimal spreading
    phi1 = (1 + jnp.sqrt(5)) / 2  # Golden ratio ≈ 1.618
    phi2 = (1 + jnp.sqrt(2))      # Silver ratio ≈ 2.414
    phi3 = jnp.pi                 # π ≈ 3.142
    
    # Generate angles using these irrational number factors
    angle1 = 2 * jnp.pi * (indices / phi1 % 1)
    angle2 = 2 * jnp.pi * (indices / phi2 % 1)
    angle3 = 2 * jnp.pi * (indices / phi3 % 1)
    
    # Convert directly to 4D coordinates (hyperspherical coordinates)
    cos_angle1, sin_angle1 = jnp.cos(angle1), jnp.sin(angle1)
    cos_angle2, sin_angle2 = jnp.cos(angle2), jnp.sin(angle2)
    cos_angle3, sin_angle3 = jnp.cos(angle3), jnp.sin(angle3)
    
    # Compute quaternion components
    w = cos_angle1
    x = sin_angle1 * cos_angle2
    y = sin_angle1 * sin_angle2 * cos_angle3
    z = sin_angle1 * sin_angle2 * sin_angle3
    
    # Combine into quaternions [w, x, y, z]
    quaternions = jnp.column_stack((w, x, y, z))
    
    # Ensure quaternions have w ≥ 0 to eliminate duplicate rotations
    # (since q and -q represent the same rotation)
    quaternions = jnp.where(
        jnp.repeat(quaternions[:, 0:1] < 0, 4, axis=1),
        -quaternions,
        quaternions
    )
    
    # Normalize (protects against floating-point errors)
    quaternions = quaternions / jnp.linalg.norm(quaternions, axis=1, keepdims=True)
    
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
def spherical_harmonic_distance(signal1, signal2):
    return jnp.linalg.norm(signal1.array - signal2.array)


@jax.jit
def spherical_grid_distance(signal1, signal2):
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
    return spherical_grid_distance(rotated_signal1, signal2)
    # return spherical_harmonic_distance(rotated_signal1, signal2)


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


def choose_best_quaternion(signal1, signal2, quaternions=None, num_quaternions=100):
    if quaternions is None:
        quaternions = evenly_distributed_quaternions(num_quaternions)
    return quaternions[jnp.argmin(jnp.array([loss_fn(q, signal1, signal2) for q in quaternions]))]


# def align_signals(
#     signal1, 
#     signal2, 
#     initial_quaternion,
#     learning_rate=1e-2,
#     num_iterations=250,
# ):
#     """Find optimal rotation to align two spherical signals using gradient descent.
    
#     Uses Adam optimizer to find the quaternion rotation that best aligns signal1
#     with signal2 by minimizing their spherical distance.
    
#     Args:
#         signal1: First spherical signal to be rotated
#         signal2: Second spherical signal (target)
#         initial_quaternion: Starting quaternion for optimization
#         learning_rate: Learning rate for Adam optimizer
#         num_iterations: Number of optimization iterations
        
#     Returns:
#         jnp.ndarray: Optimized quaternion that best aligns the signals
#     """
#     # Create optimizer
#     optimizer = optax.adam(learning_rate)
    
#     @jax.jit
#     def update_step(params, opt_state, signal1, signal2):
#         grads = jax.grad(loss_fn)(params, signal1, signal2)
#         updates, opt_state = optimizer.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#         params = normalize_quaternion(params)
#         return params, opt_state

#     # Initialize optimization
#     current_quaternion = initial_quaternion
#     opt_state = optimizer.init(current_quaternion)
    
#     # Run optimization
#     for _ in range(num_iterations):
#         current_quaternion, opt_state = update_step(
#             current_quaternion, opt_state, signal1, signal2
#         )
        
#     final_loss = loss_fn(current_quaternion, signal1, signal2)

#     return current_quaternion, final_loss

# def align_signals(
#     signal1, 
#     signal2, 
#     initial_quaternion,
#     learning_rate=1e-2,
#     num_iterations=250,
#     patience=100,
# ):
#     """Find optimal rotation to align two spherical signals using gradient descent.
    
#     Uses Adam optimizer to find the quaternion rotation that best aligns signal1
#     with signal2 by minimizing their spherical distance.
    
#     Args:
#         signal1: First spherical signal to be rotated
#         signal2: Second spherical signal (target)
#         initial_quaternion: Starting quaternion for optimization
#         learning_rate: Learning rate for Adam optimizer
#         num_iterations: Number of optimization iterations
#         patience: Number of iterations to wait before early stopping if no improvement
        
#     Returns:
#         jnp.ndarray: Optimized quaternion that best aligns the signals
#         float: Final loss value
#     """
#     # Create optimizer
#     optimizer = optax.adam(learning_rate)
    
#     @jax.jit
#     def update_step(params, opt_state, signal1, signal2):
#         grads = jax.grad(loss_fn)(params, signal1, signal2)
#         updates, opt_state = optimizer.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#         params = normalize_quaternion(params)
#         return params, opt_state, loss_fn(params, signal1, signal2)

#     # Initialize optimization
#     current_quaternion = initial_quaternion
#     opt_state = optimizer.init(current_quaternion)
    
#     # Initialize variables for early stopping
#     best_loss = float('inf')
#     best_quaternion = current_quaternion
#     patience_counter = 0
    
#     # Run optimization
#     for i in range(num_iterations):
#         current_quaternion, opt_state, current_loss = update_step(
#             current_quaternion, opt_state, signal1, signal2
#         )
        
#         # Check if loss improved
#         if current_loss < best_loss:
#             best_loss = current_loss
#             best_quaternion = current_quaternion
#             patience_counter = 0
#         else:
#             patience_counter += 1
            
#         # Early stopping check
#         if patience_counter >= patience:
#             print(f"Early stopping at iteration {i} due to no improvement for {patience} iterations")
#             break
            
#     return best_quaternion, best_loss


def align_signals(
    signal1, 
    signal2, 
    initial_quaternion,
    learning_rate=1e-2,
    num_iterations=250,
    patience=100,
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
        patience: Number of iterations to wait before early stopping if no improvement
        
    Returns:
        jnp.ndarray: Optimized quaternion that best aligns the signals
        float: Final loss value
        list: List of quaternions at each iteration
    """
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    
    @jax.jit
    def update_step(params, opt_state, signal1, signal2):
        grads = jax.grad(loss_fn)(params, signal1, signal2)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = normalize_quaternion(params)
        return params, opt_state, loss_fn(params, signal1, signal2)

    # Initialize optimization
    current_quaternion = initial_quaternion
    opt_state = optimizer.init(current_quaternion)
    
    # Initialize variables for early stopping
    best_loss = float('inf')
    best_quaternion = current_quaternion
    patience_counter = 0
    
    # Track quaternions at each iteration
    quaternion_history = [initial_quaternion.copy()]
    
    # Run optimization
    for i in range(num_iterations):
        current_quaternion, opt_state, current_loss = update_step(
            current_quaternion, opt_state, signal1, signal2
        )
        
        # Store current quaternion
        quaternion_history.append(current_quaternion.copy())
        
        # Check if loss improved
        if current_loss < best_loss:
            best_loss = current_loss
            best_quaternion = current_quaternion
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping at iteration {i} due to no improvement for {patience} iterations")
            break
            
    return best_quaternion, best_loss, quaternion_history



def rotate_points_quaternion(q, points):
    """
    Rotate 3D points using a quaternion.
    
    Parameters:
    q : array-like
        Quaternion in format [w, x, y, z] where w is the scalar component
        and [x, y, z] is the vector component. Must be a unit quaternion.
    points : array-like
        Array of points with shape (N, 3) where N is the number of points
        and each point is [x, y, z]
        
    Returns:
    jax.numpy.ndarray
        Array of rotated points with same shape as input points
    """
    # Convert inputs to jax arrays
    q = jnp.array(q, dtype=float)
    points = jnp.array(points, dtype=float)
    
    # Normalize quaternion if not already unit length
    q = q / jnp.linalg.norm(q)
    
    # Extract components
    w, x, y, z = q
    
    # Construct rotation matrix from quaternion
    R = jnp.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    # Apply rotation to all points
    rotated_points = jnp.dot(points, R.T)
    
    return rotated_points


# def stack_points(points):
#     """
#     Stack points centered about the origin that lie along the same radial line. 
#     This is used to recover the original geometry from interverting spectra.

#     Args:
#         points: Array of points with shape (N, 3) where N is the number of points
#             and each point is [x, y, z]
    
#     Returns:
#         jax.numpy.ndarray: Array of stacked points with shape (M, 3) where M is the number of stacked points
#     """
#     points_norms = jnp.linalg.norm(points, axis=1, keepdims=True)
#     normalized_points = points / points_norms

#     point_stacked_indices = []
#     indices_left = set(range(len(points)))
#     while len(indices_left):
#         point = normalized_points[next(iter(indices_left))]
#         stacked_indices = jnp.where(jnp.dot(normalized_points, point) > 0.866)[0]
#         point_stacked_indices.append(stacked_indices)
#         indices_left = indices_left - set(stacked_indices.tolist())

#     original_points = []
#     for stacked_indices in point_stacked_indices:
#         normalized_point = normalized_points[stacked_indices[0]]
#         norm = points_norms[stacked_indices].sum()
#         original_points.append(normalized_point * norm)

#     # Filter out noisy points with norm < 1
#     filtered_points = []
#     for point in original_points:
#         if jnp.linalg.norm(point) >= 0.75:
#             filtered_points.append(point)

#     return jnp.array(filtered_points)



def stack_points(points):
    """
    Vectorially stack points centered about the origin that lie along the same radial line. 
    This is used to recover the original geometry from interverting spectra.

    Args:
        points: Array of points with shape (N, 3) where N is the number of points
            and each point is [x, y, z]
    
    Returns:
        jax.numpy.ndarray: Array of stacked points with shape (M, 3) where M is the number of stacked points
    """
    points_norms = jnp.linalg.norm(points, axis=1, keepdims=True)
    normalized_points = points / points_norms

    point_stacked_indices = []
    indices_left = set(range(len(points)))
    while len(indices_left):
        point = normalized_points[next(iter(indices_left))]
        stacked_indices = jnp.where(jnp.dot(normalized_points, point) > 0.866)[0]
        point_stacked_indices.append(stacked_indices)
        indices_left = indices_left - set(stacked_indices.tolist())

    original_points = []
    for stacked_indices in point_stacked_indices:
        original_points.append(jnp.sum(points[stacked_indices], axis=0))

    # Filter out noisy points with norm < 1
    filtered_points = []
    max_point_norm = jnp.max(points_norms)
    for point in original_points:
        if jnp.linalg.norm(point) >= 0.5 * max_point_norm:
            filtered_points.append(point)

    return jnp.array(filtered_points)




def point_distance(points1: jnp.ndarray, points2: jnp.ndarray) -> float:
    """Calculate the sum of minimum distances from each point in points1 to points2.
    
    This function computes an asymmetric distance measure between two sets of points,
    typically used when points1 represents predicted/perturbed points and points2
    represents ground truth points. For each point in points1, it finds the distance
    to its closest point in points2 and sums these minimum distances.

    Args:
        points1: Array of shape (N, D) representing N predicted/perturbed points in D dimensions.
               Each row represents a point.
        points2: Array of shape (M, D) representing M ground truth points in D dimensions.
               Each row represents a point.
    
    Returns:
        tuple: A tuple containing:
            float: Sum of the minimum distances from each point in points1 to any point
                in points2. A smaller value indicates better alignment between the point sets.
            float: Maximum of the minimum distances from each point in points1 to any point
                in points2. A smaller value indicates better alignment between the point sets.
    """
    # Compute pairwise distances between all points
    distances = jnp.linalg.norm(
        points1[:, None, :] - points2[None, :, :],
        axis=-1
    )
    
    # Find minimum distances from each point in points1 to points2
    min_distances = jnp.min(distances, axis=1)
    
    # Return sum and max of all minimum distances
    return jnp.sum(min_distances), jnp.max(min_distances)


