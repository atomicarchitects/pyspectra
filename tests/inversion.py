import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

from typing import Optional, Union, Callable, Tuple  # Type hints
import e3nn_jax as e3nn  # E(3)-equivariant neural networks
import jax  # Accelerated numerical computing and automatic differentiation
import optax  # Gradient-based optimization
import jax.numpy as jnp  # Compatibility with JAX functions
import matplotlib.pyplot as plt  # Data visualization
from matplotlib.lines import Line2D  # Custom line styles
import plotly  # Interactive data visualization
import plotly.graph_objects as go  # Interactive plots
import pandas as pd  # Data manipulation and analysis
from tqdm import tqdm  # Progress bars and iteration monitoring
from src.spectra import Spectra, radial_cutoff, voronoi_cutoff

true_geometry = jnp.array([
    [1, 0, 0],
    [-0.5, jnp.sqrt(3)/2, 0],
    [-0.5, -jnp.sqrt(3)/2, 0]
])

bispectrum = Spectra(lmax=4, order=2)
true_spectrum = bispectrum.compute_geometry(true_geometry)


params = bispectrum.invert(true_spectrum)