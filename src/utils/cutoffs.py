from pymatgen.analysis.local_env import get_neighbors_of_site_with_index, CrystalNN
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