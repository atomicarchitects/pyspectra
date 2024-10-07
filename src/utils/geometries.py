import jax.numpy as jnp

# Linear
linear = jnp.array([
    [1, 0, 0],
    [-1, 0, 0]
])

# Trigonal planar
trigonal_planar = jnp.array([
    [1, 0, 0],
    [-0.5, jnp.sqrt(3)/2, 0],
    [-0.5, -jnp.sqrt(3)/2, 0]
])

# Bent (assume 120 degrees)
bent_120 = jnp.array([
    [1, 0, 0],
    [-0.5, jnp.sqrt(3)/2, 0]
])

# Tetrahedral
tetrahedral = jnp.array([
    [0, 0, 1],
    [2 * jnp.sqrt(2) / 3, 0, -1 / 3],
    [-jnp.sqrt(2) / 3, jnp.sqrt(6) / 3, -1 / 3],
    [-jnp.sqrt(2) / 3, -jnp.sqrt(6) / 3, -1 / 3]
])

# Seesaw
seesaw = jnp.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# T-shaped
t_shaped = jnp.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0]
])

# Octahedral
octahedral = jnp.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1]
])

trigonal_prism = jnp.array([
    [1, 0, 0],
    [-0.5, jnp.sqrt(3)/2, 0],
    [-0.5, -jnp.sqrt(3)/2, 0],
    [1, 0, 1],
    [-0.5, jnp.sqrt(3)/2, 1],
    [-0.5, -jnp.sqrt(3)/2, 1]
])

one_neighbor = jnp.array([
    [1, 0, 0]
])

common_geometries = {
    'linear': linear,
    'trigonal_planar': trigonal_planar,
    'bent_120': bent_120,
    'tetrahedral': tetrahedral,
    'seesaw': seesaw,
    't_shaped': t_shaped,
    'octahedral': octahedral,
    'trigonal_prism': trigonal_prism,
    'one_neighbor': one_neighbor
}

