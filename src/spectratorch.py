import e3nn
from e3nn import o3, io
from pymatgen.core.structure import Structure
import numpy as np


class SpectraTorch:
    def __init__(self, lmax: int, order: int = 2, neighbors: None = None, cutoff: callable = None):
        self.lmax = lmax
        self.order = order # 1 = power spectrum, 2 = bispectrum, 3 = trispectrum
        if neighbors is None:
            neighbors = []
        self.neighbors = neighbors
        self.cutoff = cutoff

        sph = io.SphericalTensor(lmax, p_val=1, p_arg=-1)
        self.sph = sph

        powerspectrum_ = o3.ReducedTensorProducts(
            'ij=ji', i=sph, 
            filter_ir_out=['0e', '0o'], 
            filter_ir_mid=o3.Irrep.iterator(lmax)
        )
        powerspectrum = lambda x: powerspectrum_(x, x)

        bispectrum_ = o3.ReducedTensorProducts(
            'ijk=jik=ikj', i=sph, 
            filter_ir_out=['0e', '0o'], 
            filter_ir_mid=o3.Irrep.iterator(lmax)
        )
        bispectrum = lambda x: bispectrum_(x, x, x)

        self.spectrum_function = {1: powerspectrum, 2: bispectrum}[order]

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
        local_env = [atom for atom in local_env if atom.specie.symbol in self.neighbors]
        if not local_env:
            return None
        local_env = np.stack([atom.coords for atom in local_env], axis=0) - structure[atom_site_number].coords.reshape(1, 3)

        # compute the spectra of the local environment
        sh_expansion = self.sph.with_peaks_at(local_env, self.lmax)
        return self.spectrum_function(sh_expansion)