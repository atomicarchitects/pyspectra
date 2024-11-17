import jax
from jax import lax
import e3nn_jax as e3nn
import jax.numpy as jnp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser
import optax
import chex
import matplotlib as plt
from utils.cutoffs import crystalnn_cutoff
from utils.elements import get_element


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


# def with_peaks_at(vectors: chex.Array, lmax: int, use_sum_of_diracs: bool = True) -> e3nn.IrrepsArray:
#     """
#     Compute a spherical harmonics expansion given Dirac delta functions defined on the sphere.

#     Parameters:
#         vectors (jnp.ndarray): An array of vectors. Each row represents a vector.
#         lmax (int): The maximum degree of the spherical harmonics expansion.

#     Returns:
#         e3nn.IrrepsArray: An array representing the weighted sum of the spherical harmonics expansion.
#     """
#     if use_sum_of_diracs:
#         return sum_of_diracs(vectors, lmax)

#     values = jnp.linalg.norm(vectors, axis=1)

#     mask = (values != 0)
#     vectors = jnp.where(mask[:, None], vectors, 0)
#     values = jnp.where(mask, values, 0)
 
#     coeff = e3nn.spherical_harmonics(e3nn.s2_irreps(lmax), e3nn.IrrepsArray("1o", vectors), normalize=True).array
    
#     A = jnp.einsum(
#         "ai,bi->ab",
#         coeff,
#         coeff
#     )
#     solution = jnp.linalg.lstsq(A, values)[0]
    
#     assert jnp.max(jnp.abs(values - A @ solution)) < 1e-5 * jnp.max(jnp.abs(values))

#     if jnp.max(jnp.abs(values - A @ solution)) < 1e-5 * jnp.max(jnp.abs(values)):
#         return sum_of_diracs(vectors, lmax)

#     sh_expansion = solution @ coeff
    
#     irreps = e3nn.s2_irreps(lmax)
    
#     return e3nn.IrrepsArray(irreps, sh_expansion)


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




class Spectra:
    """
    The Spectra class is used to compute the power spectrum, bispectrum, and trispectrum of an array of irreducible representations.
    It also provides methods to set the maximum degree of spherical harmonics, the order of the spectrum, and the neighbors to consider.
    """
    def __init__(self, lmax: int = 4, order: int = 2, cutoff: callable = crystalnn_cutoff):
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
        # if neighbors is None:
        #     neighbors = []
        # self.neighbors = neighbors
        self.cutoff = cutoff
        self.spectrum_function = {1: powerspectrum, 2: bispectrum, 3: trispectrum}[order]  # revise this to combine it with lmax
        self.structure = None


    def __str__(self) -> str:
        return f"Spectra(lmax={self.lmax}, order={self.order}, cutoff={self.cutoff.__name__})"


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


    # def set_neighbors(self, neighbors):
    #     """
    #     Sets the neighbors to consider.

    #     Parameters:
    #         neighbors (list): The neighbors to consider.

    #     Returns:
    #         None
    #     """
    #     self.neighbors = neighbors


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
        return self.structure.composition.reduced_formula


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


    def get_local_environment(self, site_index, inclusive_neighbors=None, exact_neighbors=None):
        """
        Gets the neighbors of a given atom.

        Parameters:
            site_index (int): The atom site number.

        Returns:
            list: The neighbors of the given atom.
        """
        neighbors = self.cutoff(self.structure, site_index)
        if neighbors:
            if inclusive_neighbors is not None:
                for atom in neighbors:
                    if get_element(atom) in inclusive_neighbors:
                        return neighbors
                return None
            if exact_neighbors is not None:
                neighbors = [atom for atom in neighbors if get_element(atom) in exact_neighbors]
            return neighbors
        return None


    def get_local_geometry(self, site_index, inclusive_neighbors=None, exact_neighbors=None):
        """
        Gets the local environment of a given atom.

        Parameters:
            site_index (int): The atom site number.

        Returns:
            list: The local environment of the given atom.
        """
        local_env = self.get_local_environment(site_index, inclusive_neighbors, exact_neighbors)
        return jnp.stack([atom.coords for atom in local_env], axis=0) - self.structure[site_index].coords.reshape(1, 3) if local_env else None


    def get_local_neighbors(self, site_index):
        """
        Gets the elements of the local environment of a given atom.

        Parameters:
            site_index (int): The atom site number.

        Returns:
            list: The elements of the local environment of the given atom.
        """
        local_env = self.get_local_environment(site_index)
        return [get_element(atom) for atom in local_env] if local_env else None

        # if len(local_env) > 0:
        #     if len(self.neighbors) > 0:
        #         local_env = [atom for atom in local_env if get_element(atom) in self.neighbors]
        #     return [get_element(atom) for atom in local_env] if local_env else None
        # return None
    

    def compute_atom_spectra(self, site_index):
        """
        Computes the spectra of the local environment of a single atom.

        Parameters:
            site_index (int): The atom site number.

        Returns:
            dict: A dictionary mapping atom site number to the spectrum of that atom's local environment.
        """
        local_geometry = self.get_local_geometry(site_index)
        if local_geometry is not None:
            signal = sum_of_diracs(local_geometry, self.lmax)
            return self.spectrum_function(signal)
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


    def compute_element_spectra(self, element, inclusive_neighbors=None, exact_neighbors=None): #TODO: inclusive and exclusive neighbors
        """
        Computes the spectra of all atoms of a given element in a structure.

        Parameters:
            element (str): The element.

        Returns:
            dict: A dictionary mapping atom site number to the spectrum of that atom's local environment.
        """
        element_indices = self.get_element_indices(element)
        ans = {}
        for site_index in element_indices:
            neighbors = self.get_local_neighbors(site_index)

            if exact_neighbors is not None and set(neighbors) != set(exact_neighbors):
                continue

            if inclusive_neighbors is not None and not set(inclusive_neighbors).issubset(set(neighbors)):
                continue

            atom_spectra = self.compute_atom_spectra(site_index)
            if atom_spectra is not None:
                ans[site_index] = atom_spectra
        return ans


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
        initial_geometry = icosahedron + 0.1 * noise
        parameters = {"predicted_geometry": initial_geometry}
        optimizer = optax.adam(learning_rate=1e-2)

        optimizer_state = optimizer.init(parameters)
        
        @jax.jit
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
                print(f"Iteration {i}, Loss: {loss_value.item()}")

        return parameters["predicted_geometry"] # TODO: use find_peaks

