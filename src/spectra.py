import e3nn_jax as e3nn
import jax.numpy as jnp
from pymatgen.core.structure import Structure


# TODO: remove this function once e3nn_jax is updated
def with_peaks_at(vectors, lmax):
    """
    Compute a spherical harmonics expansion given Dirac delta functions defined on the sphere.

    Parameters:
        vectors (jnp.ndarray): An array of vectors. Each row represents a vector.
        lmax (int): The maximum degree of the spherical harmonics expansion.

    Returns:
        e3nn.IrrepsArray: An array representing the weighted sum of the spherical harmonics expansion.
    """
    values = jnp.linalg.norm(vectors, axis=1)

    mask = values != 0
    vectors = jnp.where(mask[:, None], vectors, 0)
    values = jnp.where(mask, values, 0)
 
    coeff_list = [e3nn.spherical_harmonics(i, e3nn.IrrepsArray("1o", vectors), normalize=True).array for i in range(lmax + 1)]
    coeff = jnp.concatenate(coeff_list, axis=1)
    
    A = jnp.einsum(
        "ai,bi->ab",
        coeff,
        coeff
    )
    solution = jnp.array(jnp.linalg.lstsq(A, values)[0])  
    assert jnp.max(jnp.abs(values - A @ solution)) < 1e-5 * jnp.max(jnp.abs(values))

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
        self.spectrum_function = {1: powerspectrum, 2: bispectrum, 3: trispectrum}[order]

    def compute(self, cif, atom_site_number):
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





    