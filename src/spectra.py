import jax
from jax import lax
import e3nn_jax as e3nn
import jax.numpy as jnp
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index
import optax
import chex


def sum_of_diracs(vectors: chex.Array, lmax: int) -> e3nn.IrrepsArray:
    """Returns the norm-weighted sum of Dirac delta functions defined on the sphere."""
    values = jnp.linalg.norm(vectors, axis=1)
    return e3nn.sum(e3nn.s2_dirac(vectors, lmax, p_val=1, p_arg=-1) * values[:, None], axis=0)


# TODO: remove this function once e3nn_jax is updated
def with_peaks_at(vectors: chex.Array, lmax: int, use_sum_of_diracs: bool = True) -> e3nn.IrrepsArray:
    """
    Compute a spherical harmonics expansion given Dirac delta functions defined on the sphere.

    Parameters:
        vectors (jnp.ndarray): An array of vectors. Each row represents a vector.
        lmax (int): The maximum degree of the spherical harmonics expansion.

    Returns:
        e3nn.IrrepsArray: An array representing the weighted sum of the spherical harmonics expansion.
    """
    if use_sum_of_diracs:
        return sum_of_diracs(vectors, lmax)

    values = jnp.linalg.norm(vectors, axis=1)

    mask = (values != 0)
    vectors = jnp.where(mask[:, None], vectors, 0)
    values = jnp.where(mask, values, 0)
 
    coeff = e3nn.spherical_harmonics(e3nn.s2_irreps(lmax), e3nn.IrrepsArray("1o", vectors), normalize=True).array
    
    A = jnp.einsum(
        "ai,bi->ab",
        coeff,
        coeff
    )
    solution = jnp.linalg.lstsq(A, values)[0]
    
    # assert jnp.max(jnp.abs(values - A @ solution)) < 1e-5 * jnp.max(jnp.abs(values)) # checkify

    sh_expansion = solution @ coeff
    
    irreps = e3nn.s2_irreps(lmax)
    
    return e3nn.IrrepsArray(irreps, sh_expansion)


def powerspectrum(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    """
    Computes the power spectrum given an array of irreps.

    Parameters:
        x (e3nn.IrrepsArray): Input array of irreducible representations.

    Returns:
        e3nn.IrrepsArray: The power spectrum of the input array.
    """
    rtp = e3nn.reduced_symmetric_tensor_product_basis(x.irreps, 2, keep_ir=['0o', '0e'])
    return e3nn.IrrepsArray(rtp.irreps, jnp.einsum("i,j,ijz->z", x.array, x.array, rtp.array)).array


def bispectrum(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    """
    Computes the bispectrum given an array of irreps.

    Parameters:
        x (e3nn.IrrepsArray): Input array of irreps.

    Returns:
        e3nn.IrrepsArray: The bispectrum of the input array.
    """
    rtp = e3nn.reduced_symmetric_tensor_product_basis(x.irreps, 3, keep_ir=['0o', '0e'])
    return e3nn.IrrepsArray(rtp.irreps, jnp.einsum("i,j,k,ijkz->z", x.array, x.array, x.array, rtp.array)).array


def trispectrum(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    """
    Computes the trispectrum given an array of irreps.

    Parameters:
        x (e3nn.IrrepsArray): Input array of irreps.

    Returns:
        e3nn.IrrepsArray: The trispectrum of the input array.
    """
    rtp = e3nn.reduced_symmetric_tensor_product_basis(x.irreps, 4, keep_ir=['0o', '0e'])
    return e3nn.IrrepsArray(rtp.irreps, jnp.einsum("i,j,k,l,ijklz->z", x.array, x.array, x.array, x.array, rtp.array)).array


class Spectra:
    def __init__(self, lmax: int, order: int = 2, neighbors: None = None, cutoff: callable = None):
        self.lmax = lmax
        self.order = order # 1 = power spectrum, 2 = bispectrum, 3 = trispectrum
        if neighbors is None:
            neighbors = []
        self.neighbors = neighbors
        self.cutoff = cutoff
        self.spectrum_function = {1: powerspectrum, 2: bispectrum, 3: trispectrum}[order] # revise this to combinte it with lmax


    def compute_geometry(self, geometry):
        """
        TODO: this is a really bad name, it computes the spectrum not the geometry
        """
        sh_expansion = with_peaks_at(geometry, self.lmax)
        return self.spectrum_function(sh_expansion)


    def compute_cif_file_atom(self, cif, atom_site_number):
        """
        Compute the spectra of the local environment of a single atom in a CIF file.

        Parameters:
            cif (str): The path to the CIF file.
            atom_site_number (int): The atom site number.

        Returns:
            dict: A dictionary mapping atom site number to the spectrum of that atom's local environment.
        """
        
        # load the CIF file
        structure = Structure.from_file(cif)

        # compute the local environment of the atom
        local_env = self.cutoff(structure, atom_site_number)
        if self.neighbors:
            local_env = [atom for atom in local_env if atom.specie.symbol in self.neighbors]
        if not local_env:
            return None
        local_env = jnp.stack([atom.coords for atom in local_env], axis=0) - structure[atom_site_number].coords.reshape(1, 3)

        # compute the spectra of the local environment
        sh_expansion = with_peaks_at(local_env, self.lmax)
        return self.spectrum_function(sh_expansion)


    def invert(self, true_spectrum, n_points=12):
        """
        Invert the spectra to obtain the original local environment up to a rotation.
        """
        rng = jax.random.PRNGKey(0)
        init_rng, rng = jax.random.split(rng)
        init_geometry = jax.random.normal(init_rng, (n_points, 3))
        init_geometry /= jnp.linalg.norm(init_geometry, axis=1, keepdims=True)
        init_params = {"predicted_geometry": init_geometry}
        optimizer = optax.adam(learning_rate=1e-2)
        return self.fit(init_params, optimizer, true_spectrum)


    def fit(
            self, 
            params: optax.Params, 
            optimizer: optax.GradientTransformation, 
            true_spectrum: jnp.ndarray,
            max_iter: int = 2000
        ) -> optax.Params:
        """
        Performs fitting on the provided parameters.
        
        Args:
            params (optax.Params): Initial parameters.
            optimizer (optax.GradientTransformation): The optimizer to use.
            true_spectrum (jnp.ndarray): True power spectrum.
            max_iter (int, optional): Maximum number of iterations. Defaults to 2500.

        Returns:
            optax.Params: The fitted parameters.
        """
        opt_state = optimizer.init(params)
        
        def loss(params):
            predicted_geometry = params["predicted_geometry"]
            predicted_signal = with_peaks_at(predicted_geometry, self.lmax)
            predicted_spectrum = self.spectrum_function(predicted_signal)
            return jnp.abs(true_spectrum - predicted_spectrum).mean()

        @jax.jit
        def step(params, opt_state):
            loss_value, grads = jax.value_and_grad(loss)(params)
            grad_norms = jnp.linalg.norm(grads["predicted_geometry"], axis=1)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value, grad_norms

        all_steps = []
        all_params = []
        all_losses = []
        all_grad_norms = []
        for iter in range(max_iter):
            params, opt_state, loss_value, grad_norms = step(params, opt_state)
            if iter % 100 == 0:
                print(f"Step {iter}, Loss: {loss_value}")
            
            if iter % 10 == 0:
                all_steps.append(iter)
                all_params.append(params)
                all_losses.append(loss_value)
                all_grad_norms.append(grad_norms)

        return all_steps, all_params, all_losses, all_grad_norms


def radial_cutoff(radius=3.0):
    """
    This function creates a cutoff function for a given radius.

    Parameters:
        radius (float): The cutoff radius. Default is 3.0.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff radius of the given atom.
    """
    return lambda structure, atom: structure.get_neighbors(structure[int(atom)], radius)


def voronoi_cutoff(tol, cutoff): # add defaults
    """
    This function creates a cutoff function using the Voronoi approach.

    Parameters:
        tol (float): The tolerance for the Voronoi calculation.
        cutoff (float): The cutoff distance for the Voronoi calculation.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff distance of the given atom using the Voronoi approach.
    """
    return lambda structure, atom: get_neighbors_of_site_with_index(structure, int(atom), approach="voronoi", tol=tol, cutoff=cutoff)


def min_dist_cutoff(tol, cutoff):
    """
    This function creates a cutoff function using the minimum distance approach.

    Parameters:
        tol (float): The tolerance for the minimum distance calculation.
        cutoff (float): The cutoff distance for the minimum distance calculation.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff distance of the given atom using the minimum distance approach.
    """
    return lambda structure, atom: get_neighbors_of_site_with_index(structure, int(atom), approach="min_dist", tol=tol, cutoff=cutoff)




