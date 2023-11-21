import jax
from jax import lax
import e3nn_jax as e3nn
import jax.numpy as jnp
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
    LightStructureEnvironments,
)
import optax
import chex
import plotly
import plotly.graph_objects as go


def sum_of_diracs(vectors: chex.Array, lmax: int, values: chex.Array = None) -> e3nn.IrrepsArray:
    """
    Given a set of vectors, computes the sum of Dirac delta functions.

    Parameters:
        vectors (chex.Array): Input array of vectors.
        lmax (int): Maximum degree of spherical harmonics.
        values (chex.Array, optional): Values at each vector. If not provided, the norm of each vector is used.

    Returns:
        e3nn.IrrepsArray: The sum of Dirac delta functions.
    """
    if values is None:
        values = jnp.linalg.norm(vectors, axis=1)
    return e3nn.sum(e3nn.s2_dirac(vectors, lmax, p_val=1, p_arg=-1) * values[:, None], axis=0)


def powerspectrum(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    """
    Computes the power spectrum of an array of irreducible representations.

    Parameters:
        x (e3nn.IrrepsArray): Input array of irreducible representations.

    Returns:
        e3nn.IrrepsArray: The power spectrum of the input array.
    """
    rtp = e3nn.reduced_symmetric_tensor_product_basis(x.irreps, 2, keep_ir=['0o', '0e'])
    return e3nn.IrrepsArray(rtp.irreps, jnp.einsum("i,j,ijz->z", x.array, x.array, rtp.array)).array


def bispectrum(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    """
    Computes the bispectrum of an array of irreducible representations.

    Parameters:
        x (e3nn.IrrepsArray): Input array of irreps.

    Returns:
        e3nn.IrrepsArray: The bispectrum of the input array.
    """
    rtp = e3nn.reduced_symmetric_tensor_product_basis(x.irreps, 3, keep_ir=['0o', '0e'])
    return e3nn.IrrepsArray(rtp.irreps, jnp.einsum("i,j,k,ijkz->z", x.array, x.array, x.array, rtp.array)).array


def trispectrum(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    """
    Computes the trispectrum of an array of irreducible representations.

    Parameters:
        x (e3nn.IrrepsArray): Input array of irreps.

    Returns:
        e3nn.IrrepsArray: The trispectrum of the input array.
    """
    rtp = e3nn.reduced_symmetric_tensor_product_basis(x.irreps, 4, keep_ir=['0o', '0e'])
    return e3nn.IrrepsArray(rtp.irreps, jnp.einsum("i,j,k,l,ijklz->z", x.array, x.array, x.array, x.array, rtp.array)).array


def radial_cutoff(radius=3.0):
    """
    This function creates a cutoff function for a given radius.

    Parameters:
        radius (float): The cutoff radius. Default is 3.0.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff radius of the given atom.
    """
    return lambda structure, atom: structure.get_neighbors(structure[int(atom)], radius)


def voronoi_cutoff(tol, cutoff):  # TODO: add defaults
    """
    This function creates a cutoff function using the Voronoi approach.

    Parameters:
        tol (float): The tolerance for the Voronoi calculation.
        cutoff (float): The cutoff distance for the Voronoi calculation.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff distance of the given atom using the Voronoi approach.
    """
    return lambda structure, atom: get_neighbors_of_site_with_index(structure, int(atom), approach="voronoi", tol=tol, cutoff=cutoff)


def min_dist_cutoff(delta=0.1, cutoff=10):
    """
    This function creates a cutoff function using the minimum distance approach.

    Parameters:
        delta (float): Tolerance involved in neighbor finding.
        cutoff (float): (Large) radius to find tentative neighbors.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff distance of the given atom using the minimum distance approach.
    """
    return lambda structure, atom: get_neighbors_of_site_with_index(structure, int(atom), approach="min_dist", delta=delta, cutoff=cutoff)


def chemenv_cutoff(structure, atom_site_number): 
    """
    This function creates a cutoff function using the ChemEnv approach.

    Parameters:
        structure (Structure): The structure for which the cutoff is to be calculated.
        atom_site_number (int): The site number of the atom for which the cutoff is to be calculated.

    Returns:
        list: A list of atoms within the cutoff distance of the given atom using the ChemEnv approach.
    """
    lgf = LocalGeometryFinder()
    lgf.setup_structure(structure)
    se = lgf.compute_structure_environments(structure, only_indices=[atom_site_number])
    strategy = SimplestChemenvStrategy()
    lse = LightStructureEnvironments.from_structure_environments(
        strategy=strategy, structure_environments=se
    )
    return [structure[neighbor['index']] for neighbor in lse.neighbors_sets[atom_site_number][0].neighb_sites_and_indices]


def visualize(geometry):
    sig = sum_of_diracs(geometry, lmax=4)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='', showticklabels=False, showgrid=False, zeroline=False, backgroundcolor='rgba(255,255,255,255)', range=[-2.5, 2.5]),
            yaxis=dict(title='', showticklabels=False, showgrid=False, zeroline=False, backgroundcolor='rgba(255,255,255,255)', range=[-2.5, 2.5]),
            zaxis=dict(title='', showticklabels=False, showgrid=False, zeroline=False, backgroundcolor='rgba(255,255,255,255)', range=[-2.5, 2.5]),
            bgcolor='rgba(255,255,255,255)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=0.5, y=0.5, z=0.5)
            )
        ),
        plot_bgcolor='rgba(255,255,255,255)',
        paper_bgcolor='rgba(255,255,255,255)',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=0
        )
    )

    spherical_harmonics_trace = go.Surface(e3nn.to_s2grid(sig, 100, 99, quadrature="soft").plotly_surface(radius=1., normalize_radius_by_max_amplitude=True, scale_radius_by_amplitude=True), name="Signal", showlegend=True)
    atoms_trace = go.Scatter3d(x=geometry[:, 0], y=geometry[:, 1], z=geometry[:, 2], mode='markers', marker=dict(size=10, color='black'), showlegend=True, name="Points")
    fig = go.Figure()
    fig.add_trace(spherical_harmonics_trace)
    fig.add_trace(atoms_trace)
    fig.update_layout(layout)
    return fig



class Spectra:
    """
    The Spectra class is used to compute the power spectrum, bispectrum, and trispectrum of an array of irreducible representations.
    It also provides methods to set the maximum degree of spherical harmonics, the order of the spectrum, and the neighbors to consider.
    """
    
    def __init__(self, lmax: int = 4, order: int = 2, neighbors: None = None, cutoff: callable = chemenv_cutoff):
        """
        Initializes the Spectra class.

        Parameters:
            lmax (int): The maximum degree of spherical harmonics.
            order (int): The order of the spectrum. Default is 2.
            neighbors (list): The neighbors to consider. Default is None.
            cutoff (callable): The cutoff function to use. Default is None.
        """
        self.lmax = lmax
        self.order = order  # 1 = power spectrum, 2 = bispectrum, 3 = trispectrum
        if neighbors is None:
            neighbors = []
        self.neighbors = neighbors
        self.cutoff = cutoff
        self.spectrum_function = {1: powerspectrum, 2: bispectrum, 3: trispectrum}[order]  # revise this to combine it with lmax
        self.structure = None


    def set_lmax(self, lmax):
        """
        Sets the maximum degree of spherical harmonics.

        Parameters:
            lmax (int): The maximum degree of spherical harmonics.

        Returns:
            None
        """
        self.lmax = lmax
        self.spectrum_function = {1: powerspectrum, 2: bispectrum, 3: trispectrum}[self.order]


    def set_order(self, order):
        """
        Sets the order of the spectrum.

        Parameters:
            order (int): The order of the spectrum.

        Returns:
            None
        """
        self.order = order
        self.spectrum_function = {1: powerspectrum, 2: bispectrum, 3: trispectrum}[self.order]


    def set_neighbors(self, neighbors):
        """
        Sets the neighbors to consider.

        Parameters:
            neighbors (list): The neighbors to consider.

        Returns:
            None
        """
        self.neighbors = neighbors


    def set_cutoff(self, cutoff):
        """
        Sets the cutoff function.

        Parameters:
            cutoff (callable): The cutoff function.

        Returns:
            None
        """
        self.cutoff = cutoff


    def load_cif(self, cif_file_path):
        """
        Loads a CIF file.

        Parameters:
            cif_file_path (str): The path to the CIF file to load.

        Returns:
            None
        """
        self.structure = Structure.from_file(cif_file_path)


    def load_structure(self, structure):
        """
        Loads a structure.

        Parameters:
            structure (Structure): The structure to load.

        Returns:
            None
        """
        self.structure = structure


    def compute_atom_spectra(self, atom_site_number):
        """
        Computes the spectra of the local environment of a single atom.

        Parameters:
            atom_site_number (int): The atom site number.

        Returns:
            dict: A dictionary mapping atom site number to the spectrum of that atom's local environment.
        """
        local_env = self.cutoff(self.structure, atom_site_number)
        if self.neighbors:
            local_env = [atom for atom in local_env if atom.specie.symbol in self.neighbors]
        if not local_env:
            return None
        local_env = jnp.stack([atom.coords for atom in local_env], axis=0) - self.structure[atom_site_number].coords.reshape(1, 3)
        sh_expansion = sum_of_diracs(local_env, self.lmax)
        return self.spectrum_function(sh_expansion)
    

    def compute_element_spectra(self, element):
        """
        Computes the spectra of all atoms of a given element in a structure.

        Parameters:
            element (str): The element.

        Returns:
            dict: A dictionary mapping atom site number to the spectrum of that atom's local environment.
        """
        return {atom_site_number: self.compute_atom_spectra(atom_site_number) for atom_site_number, atom in enumerate(self.structure) if atom.specie.symbol == element}


    def get_symmetry_unique_indices(self):
        """
        Returns the indices of the symmetry-unique atoms in a structure.

        Returns:
            set: A set of indices of the symmetry-unique atoms in the structure.
        """
        sga = SpacegroupAnalyzer(self.structure)
        return set(sga.get_symmetry_dataset()['equivalent_atoms'])


    def compute_structure_spectra(self, symmetry_unique_only=True):
        """
        Computes the spectra of all atoms in a structure.

        Parameters:
            symmetry_unique_only (bool): Whether to only compute the spectra for the symmetry-unique atoms in the structure.

        Returns:
            dict: A dictionary mapping atom site number to the spectrum of that atom's local environment.
        """
        if symmetry_unique_only:
            symmetry_unique_indices = self.get_symmetry_unique_indices()
            atom_site_numbers = [i for i in range(len(self.structure)) if i in symmetry_unique_indices]
        else:
            atom_site_numbers = range(len(self.structure))
        return {atom_site_number: self.compute_atom_spectra(atom_site_number) for atom_site_number in atom_site_numbers}


    def invert(self, true_spectrum, n_points=12):
        """
        Inverts the spectra to obtain the original local environment up to a rotation.
        """
        rng = jax.random.PRNGKey(0)
        init_rng, rng = jax.random.split(rng)
        # Adding noise to the initial geometry
        noise = jax.random.normal(rng, (12, 3))
        init_geometry = jnp.array([[1, 0 ,0]] * 12) + 0.0000001 * noise
        init_geometry /= jnp.linalg.norm(init_geometry, axis=1, keepdims=True)
        init_params = {"predicted_geometry": init_geometry}
        optimizer = optax.adam(learning_rate=1e-2)
        return self.fit(init_params, optimizer, true_spectrum)


    def fit(
            self, 
            params: optax.Params, 
            optimizer: optax.GradientTransformation, 
            true_spectrum: jnp.ndarray,
            max_iter: int = 1000
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
            predicted_signal = sum_of_diracs(predicted_geometry, self.lmax)
            predicted_spectrum = self.spectrum_function(predicted_signal)
            # return optax.l2_loss(true_spectrum, predicted_spectrum).mean() # 30x slower
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