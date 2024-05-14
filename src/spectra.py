import jax
from jax import lax
import e3nn_jax as e3nn
import jax.numpy as jnp
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index, CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    MultiWeightsChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
    LightStructureEnvironments,
)
from pymatgen.core.sites import PeriodicSite
import optax
import chex
import plotly.graph_objects as go
import matplotlib as plt


def sum_of_diracs(
    vectors: chex.Array, 
    lmax: int, 
    values: chex.Array = None
) -> e3nn.IrrepsArray:
    """
    Given a set of vectors, computes the sum of Dirac delta functions.

    Parameters:
        vectors (chex.Array): Input array of vectors.
        lmax (int): Maximum degree of spherical harmonics.
        values (chex.Array, optional): Values at each vector. 
            If not provided, the norm of each vector is used.

    Returns:
        e3nn.IrrepsArray: The sum of Dirac delta functions.
    """
    if values is None:
        values = jnp.linalg.norm(vectors, axis=1)
    return e3nn.sum(e3nn.s2_dirac(vectors, lmax, p_val=1, p_arg=-1) * values[:, None], axis=0)


def with_peaks_at(vectors: chex.Array, lmax: int, use_sum_of_diracs: bool = False) -> e3nn.IrrepsArray:
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
    
    assert jnp.max(jnp.abs(values - A @ solution)) < 1e-5 * jnp.max(jnp.abs(values)) # checkify

    sh_expansion = solution @ coeff
    
    irreps = e3nn.s2_irreps(lmax)
    
    return e3nn.IrrepsArray(irreps, sh_expansion)


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


def get_element(atom: PeriodicSite):
    """
    Gets the element of an atom.

    Parameters:
        atom (pymatgen.core.sites.PeriodicSite): The atom. 

    Returns:
        str: The element of the atom.
    """
    return atom.species.elements[0].symbol


def radial_cutoff(radius=3.0):
    """
    Get all neighbors to a site within a sphere of radius r. Excludes the site itself.

    Parameters:
        radius (float): The cutoff radius. Default is 3.0.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff radius of the given atom.
    """
    return lambda structure, atom: structure.get_neighbors(structure[int(atom)], radius)





def voronoi_cutoff(tol=0, cutoff=13.0):
    """
    Uses a Voronoi algorithm to determine near neighbors for each site in a structure.

    Parameters:
        tol (float): Tolerance parameter for near-neighbor finding. Faces that are smaller than tol fraction of the largest face are not included in the tessellation. Default is 0.
        cutoff (float): Cutoff radius in Angstroms to look for near-neighbor atoms. Default is 13.0.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff distance of the given atom using the Voronoi approach.
    """
    return lambda structure, atom: get_neighbors_of_site_with_index(structure, int(atom), approach="voronoi", delta=tol, cutoff=cutoff)


def min_dist_cutoff(tol=0.1, cutoff=10.0):
    """
    Determine near-neighbor sites using the nearest neighbor(s) at distance, d_min, plus all neighbors within a distance (1 + tol) * d_min, where tol is a (relative) distance tolerance parameter.

    Parameters:
        delta (float): Tolerance parameter for neighbor identification. Default is 0.1.
        cutoff (float): Cutoff radius in Angstrom to look for trial near-neighbor sites. Default is 10.0.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff distance of the given atom using the minimum distance approach.
    """
    return lambda structure, atom: get_neighbors_of_site_with_index(structure, int(atom), approach="min_dist", delta=tol, cutoff=cutoff)


def chemenv_cutoff(strategy="multi_weights"): 
    """
    This function creates a cutoff function using the ChemEnv approach.

    Parameters:
        structure (Structure): The structure for which the cutoff is to be calculated.
        site_index (int): The site number of the atom for which the cutoff is to be calculated.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff distance of the given atom using the ChemEnv approach.
    """
    if strategy == "simplest":
        strategy = SimplestChemenvStrategy()
    elif strategy == "multi_weights":
        strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
    else:
        raise ValueError("Invalid strategy")
    def cutoff_function(structure, site_index):
        lgf = LocalGeometryFinder()
        lgf.setup_structure(structure)
        se = lgf.compute_structure_environments(structure, only_indices=[site_index])
        lse = LightStructureEnvironments.from_structure_environments(
            strategy=strategy, structure_environments=se
        )
        neighbor_sets = lse.neighbors_sets[site_index]
        if len(neighbor_sets) == 0:
            return []
        else:
            return [neighbor['site'] for neighbor in neighbor_sets[0].neighb_sites_and_indices]
    return cutoff_function


def crystalnn_cutoff():
    """
    This function creates a cutoff function using the CrystalNN approach (with parameters set for pure geometric neighbor finding).

    Parameters:
        structure (Structure): The structure for which the cutoff is to be calculated.
        site_index (int): The site number of the atom for which the cutoff is to be calculated.

    Returns:
        function: A function that takes a structure and an atom and returns all atoms within the cutoff distance of the given atom using the CrystalNN approach.
    """
    def cutoff_function(structure, site_index):
        nn = CrystalNN(distance_cutoffs=None, x_diff_weight=0, porous_adjustment=False)
        local_env =  nn.get_nn_info(structure, site_index)
        return [neighbor['site'] for neighbor in local_env]
    return cutoff_function



def visualize_signal(signal):
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title='', 
                showticklabels=False, 
                showgrid=False, 
                zeroline=False, 
                backgroundcolor='rgba(255,255,255,255)', 
                range=[-2.5, 2.5]
            ),
            yaxis=dict(
                title='', 
                showticklabels=False, 
                showgrid=False, 
                zeroline=False, 
                backgroundcolor='rgba(255,255,255,255)', 
                range=[-2.5, 2.5]
            ),
            zaxis=dict(
                title='', 
                showticklabels=False, 
                showgrid=False, 
                zeroline=False, 
                backgroundcolor='rgba(255,255,255,255)', 
                range=[-2.5, 2.5]
            ),
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

    spherical_harmonics_trace = go.Surface(
        e3nn.to_s2grid(signal, 100, 99, quadrature="soft").plotly_surface(
            radius=1., 
            normalize_radius_by_max_amplitude=True, 
            scale_radius_by_amplitude=True
        ), 
        name="Signal", 
        showlegend=True
    )

    fig = go.Figure()
    fig.add_trace(spherical_harmonics_trace)
    fig.update_layout(layout)
    return fig


def visualize_geometry(geometry, show_points=False):
    signal = sum_of_diracs(geometry, lmax=4)
    fig = visualize_signal(signal)
    if show_points:
        atoms_trace = go.Scatter3d(
            x=geometry[:, 0], 
            y=geometry[:, 1], 
            z=geometry[:, 2], 
            mode='markers', 
            marker=dict(size=10, color='black'), 
            showlegend=True, 
            name="Points"
        )
        fig.add_trace(atoms_trace)
    return fig






def colorplot(arr: jnp.ndarray):
    """Plot spectra"""
    # TODO: add functionality to plot multiple spectra (properly take into account max and min)
    # TODO: add functionality to change dimensions of figure
    reshaped_arr = arr.reshape(3, 5)  # Reshape the array into 3 rows of 5 components each
    plt.figure(figsize=(15, 3))  # Adjust the figure size to accommodate the reshaped array
    plt.axis("off")
    vmax = jnp.maximum(jnp.abs(jnp.min(reshaped_arr)), jnp.max(reshaped_arr))  # Compute vmax using the reshaped array
    return plt.imshow(reshaped_arr, cmap="PuOr", vmin=-vmax, vmax=vmax)  # Plot the reshaped array






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


    def __str__(self) -> str:
        return f"Spectra(lmax={self.lmax}, order={self.order}, neighbors={self.neighbors}, cutoff={self.cutoff.__name__})"


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


    def load_cif(self, cif_path):
        """
        Loads a CIF file.

        Parameters:
            cif_path (str): The path to the CIF file to load.

        Returns:
            None
        """
        # self.structure = Structure.from_file(cif_path)
        cif_parser = CifParser(cif_path, occupancy_tolerance=1000)
        self.structure = cif_parser.get_structures(primitive=False)[0]


    def load_structure(self, structure):
        """
        Loads a structure.

        Parameters:
            structure (Structure): The structure to load.

        Returns:
            None
        """
        self.structure = structure

    def get_structure(self):
        """
        Gets the structure.

        Returns:
            Structure: The structure.
        """
        return self.structure


    def get_site_element(self, site_index):
        """
        Gets the element of a given atom.

        Parameters:
            site_index (int): The atom site number.

        Returns:
            str: The element of the given atom.
        """
        return get_element(self.structure[site_index])


    def get_formula(self):
        """
        Gets the formula of the structure.

        Returns:
            str: The formula of the structure.
        """
        return self.structure.formula


    def compute_geometry_spectra(self, geometry):
        """
        Computes the spectra of a given geometry.

        Parameters:
            geometry (list): The geometry to compute the spectra for.

        Returns:
            list: The computed spectra for the given geometry.
        """
        sh_expansion = sum_of_diracs(geometry, self.lmax)
        return self.spectrum_function(sh_expansion)


    def compute_spherical_harmonic_spectra(self, sh_signal):
        """
        Computes the spectra of a given set of spherical harmonics.

        Parameters:
            spherical_harmonics (list): The spherical harmonics to compute the spectra for.

        Returns:
            list: The computed spectra for the given set of spherical harmonics.
        """
        return self.spectrum_function(sh_signal)


    def get_local_neighbors(self, site_index):
        """
        Gets the neighbors of a given atom.

        Parameters:
            site_index (int): The atom site number.

        Returns:
            list: The neighbors of the given atom.
        """
        return self.cutoff(self.structure, site_index)


    def get_local_geometry(self, site_index):
        """
        Gets the local environment of a given atom.

        Parameters:
            site_index (int): The atom site number.

        Returns:
            list: The local environment of the given atom.
        """
        local_env = self.get_local_neighbors(site_index)
        if len(local_env) > 0 and len(self.neighbors) > 0:
            local_env = [atom for atom in local_env if get_element(atom) in self.neighbors]
        if len(local_env) > 0:
            return jnp.stack([atom.coords for atom in local_env], axis=0) - self.structure[site_index].coords.reshape(1, 3)
        return None


    def get_local_elements(self, site_index):
        """
        Gets the elements of the local environment of a given atom.

        Parameters:
            site_index (int): The atom site number.

        Returns:
            list: The elements of the local environment of the given atom.
        """
        local_env = self.cutoff(self.structure, site_index)
        if len(local_env) > 0:
            if len(self.neighbors) > 0:
                local_env = [atom for atom in local_env if get_element(atom) in self.neighbors]
            return [get_element(atom) for atom in local_env] if local_env else None
        return None
    

    def compute_atom_spectra(self, site_index):
        """
        Computes the spectra of the local environment of a single atom.

        Parameters:
            site_index (int): The atom site number.

        Returns:
            dict: A dictionary mapping atom site number to the spectrum of that atom's local environment.
        """
        local_env = self.get_local_geometry(site_index)
        if local_env is not None:
            sh_expansion = sum_of_diracs(local_env, self.lmax)
            return self.spectrum_function(sh_expansion)
        return None
    

    def get_element_indices(self, element):
        """
        Gets the indices of atoms of a given element in a structure.

        Parameters:
            element (str): The element.

        Returns:
            list: A list of indices of atoms of the given element in the structure.
        """
        return [site_index for site_index, atom in enumerate(self.structure) if get_element(atom) == element]

    def compute_element_spectra(self, element): #TODO: update this to use get_element_indices
        """
        Computes the spectra of all atoms of a given element in a structure.

        Parameters:
            element (str): The element.

        Returns:
            dict: A dictionary mapping atom site number to the spectrum of that atom's local environment.
        """
        return {site_index: self.compute_atom_spectra(site_index) for site_index, atom in enumerate(self.structure) if get_element(atom) == element} 


    def get_symmetry_unique_indices(self):
        """
        Returns the indices of the symmetry-unique atoms in a structure.

        Returns:
            set: A set of indices of the symmetry-unique atoms in the structure.
        """
        sga = SpacegroupAnalyzer(self.structure)
        if (symmetry_dataset := sga.get_symmetry_dataset()):
            return set(symmetry_dataset['equivalent_atoms'])
        else:
            return set(range(len(self.structure)))


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
            site_indices = [i for i in range(len(self.structure)) if i in symmetry_unique_indices]
        else:
            site_indices = range(len(self.structure))

        structure_spectra = {}
        for site_index in site_indices:
            atom_spectra = self.compute_atom_spectra(site_index)
            if atom_spectra is not None:
                structure_spectra[site_index] = atom_spectra
        return structure_spectra


    def invert(self, true_spectrum, max_iter=1000, print_loss=False): 
        """
        Inverts the spectra to obtain the geometry and performs fitting on the provided parameters.

        Parameters:
            true_spectrum (jnp.ndarray): The true power spectrum.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.

        Returns:
            jnp.ndarray: The predicted geometry.
        """
        rng = jax.random.PRNGKey(0)
        init_rng, rng = jax.random.split(rng)
        noise = jax.random.normal(rng, (12, 3))

        golden_ratio = (1 + jnp.sqrt(5)) / 2
        icosahedron = jnp.array([
            [-1, golden_ratio, 0],
            [1, golden_ratio, 0],
            [-1, -golden_ratio, 0],
            [1, -golden_ratio, 0],
            [0, -1, golden_ratio],
            [0, 1, golden_ratio],
            [0, -1, -golden_ratio],
            [0, 1, -golden_ratio],
            [golden_ratio, 0, -1],
            [golden_ratio, 0, 1],
            [-golden_ratio, 0, -1],
            [-golden_ratio, 0, 1]
        ]) / jnp.sqrt(1 + golden_ratio**2)  # Normalize to unit length
        initial_geometry = icosahedron + 0.01 * noise
        parameters = {"predicted_geometry": initial_geometry}
        optimizer = optax.adam(learning_rate=1e-2)

        optimizer_state = optimizer.init(parameters)
        
        def loss(parameters):
            predicted_geometry = parameters["predicted_geometry"]
            predicted_signal = sum_of_diracs(predicted_geometry, self.lmax)
            predicted_spectrum = self.spectrum_function(predicted_signal)
            return jnp.abs(true_spectrum - predicted_spectrum).mean()

        @jax.jit
        def step(parameters, optimizer_state):
            loss_value, grads = jax.value_and_grad(loss)(parameters)
            updates, optimizer_state = optimizer.update(grads, optimizer_state, parameters)
            parameters = optax.apply_updates(parameters, updates)
            return parameters, optimizer_state, loss_value

        for i in range(max_iter):
            parameters, optimizer_state, loss_value = step(parameters, optimizer_state)
            if print_loss and i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss_value}")

        return parameters["predicted_geometry"] # TODO: use find_peaks

