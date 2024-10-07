import numpy as np
import jax.numpy as jnp
import re
from pymatgen.core.sites import PeriodicSite

element_symbol_to_name = {
    "H": "Hydrogen", "He": "Helium", "Li": "Lithium", "Be": "Beryllium",
    "B": "Boron", "C": "Carbon", "N": "Nitrogen", "O": "Oxygen",
    "F": "Fluorine", "Ne": "Neon", "Na": "Sodium", "Mg": "Magnesium",
    "Al": "Aluminum", "Si": "Silicon", "P": "Phosphorus", "S": "Sulfur",
    "Cl": "Chlorine", "Ar": "Argon", "K": "Potassium", "Ca": "Calcium",
    "Sc": "Scandium", "Ti": "Titanium", "V": "Vanadium", "Cr": "Chromium",
    "Mn": "Manganese", "Fe": "Iron", "Co": "Cobalt", "Ni": "Nickel",
    "Cu": "Copper", "Zn": "Zinc", "Ga": "Gallium", "Ge": "Germanium",
    "As": "Arsenic", "Se": "Selenium", "Br": "Bromine", "Kr": "Krypton",
    "Rb": "Rubidium", "Sr": "Strontium", "Y": "Yttrium", "Zr": "Zirconium",
    "Nb": "Niobium", "Mo": "Molybdenum", "Tc": "Technetium", "Ru": "Ruthenium",
    "Rh": "Rhodium", "Pd": "Palladium", "Ag": "Silver", "Cd": "Cadmium",
    "In": "Indium", "Sn": "Tin", "Sb": "Antimony", "Te": "Tellurium",
    "I": "Iodine", "Xe": "Xenon", "Cs": "Cesium", "Ba": "Barium",
    "La": "Lanthanum", "Ce": "Cerium", "Pr": "Praseodymium", "Nd": "Neodymium",
    "Pm": "Promethium", "Sm": "Samarium", "Eu": "Europium", "Gd": "Gadolinium",
    "Tb": "Terbium", "Dy": "Dysprosium", "Ho": "Holmium", "Er": "Erbium",
    "Tm": "Thulium", "Yb": "Ytterbium", "Lu": "Lutetium", "Hf": "Hafnium",
    "Ta": "Tantalum", "W": "Tungsten", "Re": "Rhenium", "Os": "Osmium",
    "Ir": "Iridium", "Pt": "Platinum", "Au": "Gold", "Hg": "Mercury",
    "Tl": "Thallium", "Pb": "Lead", "Bi": "Bismuth", "Po": "Polonium",
    "At": "Astatine", "Rn": "Radon", "Fr": "Francium", "Ra": "Radium",
    "Ac": "Actinium", "Th": "Thorium", "Pa": "Protactinium", "U": "Uranium",
    "Np": "Neptunium", "Pu": "Plutonium", "Am": "Americium", "Cm": "Curium",
    "Bk": "Berkelium", "Cf": "Californium", "Es": "Einsteinium", "Fm": "Fermium",
    "Md": "Mendelevium", "No": "Nobelium", "Lr": "Lawrencium", "Rf": "Rutherfordium",
    "Db": "Dubnium", "Sg": "Seaborgium", "Bh": "Bohrium", "Hs": "Hassium",
    "Mt": "Meitnerium", "Ds": "Darmstadtium", "Rg": "Roentgenium", "Cn": "Copernicium",
    "Nh": "Nihonium", "Fl": "Flerovium", "Mc": "Moscovium", "Lv": "Livermorium",
    "Ts": "Tennessine", "Og": "Oganesson"
}


def get_element(atom: PeriodicSite):
    """
    Gets the element of an atom.

    Parameters:
        atom (pymatgen.core.sites.PeriodicSite): The atom. 

    Returns:
        str: The element of the atom.
    """
    return atom.species.elements[0].symbol



# first two rows + S + Se + groups 17 and 18
small_elements = {
    'H', 'He',                                  # First row
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',  # Second row
    'S', 'Se',                                  # Sulfur and Selenium
    'Cl', 'Br', 'I', 'At',                      # Group 17 (Halogens)
    'Ar', 'Kr', 'Xe', 'Rn'                      # Group 18 (Noble gases)
}

transition_metals_by_series = {
    '3d': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'],
    '4d': ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd'],
    '5d': ['Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'],
    '4f': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho','Er', 'Tm', 'Yb', 'Lu'],
    '5f': ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
}
transition_metal_series = transition_metals_by_series.keys()
transition_metals = set()
for series, metals in transition_metals_by_series.items():
    transition_metals.update(metals)


def contains_element(formula, element):
    return element in re.findall('([A-Z][a-z]*)', formula)


def num_elements(formula):
    return len(set(re.findall('([A-Z][a-z]*)', formula)))


def get_transition_metal_series(elements):
    transition_metal_series = set()
    for series, metals in transition_metals_by_series.items():
        if any(metal in elements for metal in metals):
            transition_metal_series.add(series)
    return transition_metal_series


def string_to_array(s, sep, shape):
    formatted_str = s.replace('\n', '').replace('[', '').replace(']', '')
    array = np.fromstring(formatted_str, sep=sep)
    return array.reshape(shape)



