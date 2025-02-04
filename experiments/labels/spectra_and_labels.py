import sys
import os
import argparse
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    MultiWeightsChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
    LightStructureEnvironments,
)
from matminer.featurizers.site import ChemEnvSiteFingerprint

project_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
src_dir = os.path.join(project_root, 'src')
sys.path.append(src_dir)

import spectra as spectra
from utils.cutoffs import chemenv_cutoff

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)
args = parser.parse_args()

random.seed(42)
strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
csf = ChemEnvSiteFingerprint.from_preset("multi_weights")
labels = csf.feature_labels()

cutoff = chemenv_cutoff(strategy='multi_weights')

df = pd.DataFrame(
    columns=[
        'cif_name',
        'site_index',
        'site_element',
        'local_elements',
        'local_geometry',
        'sh_signal_lmax_2',
        'sh_signal_lmax_3',
        'sh_signal_lmax_4',
        'sh_signal_lmax_5',
        'sh_signal_lmax_6',
        'power_spectrum_lmax_2',
        'power_spectrum_lmax_3',
        'power_spectrum_lmax_4',
        'power_spectrum_lmax_5',
        'power_spectrum_lmax_6',
        'bispectrum_lmax_2',
        'bispectrum_lmax_3',
        'bispectrum_lmax_4',
        'bispectrum_lmax_5',
        'bispectrum_lmax_6',
        'trispectrum_lmax_2',
        'trispectrum_lmax_3',
        'trispectrum_lmax_4',
        'label_vector',
        'top_label'
    ]
)

cif_dir = os.path.join(project_root, "data/mp/cifs/")
cif_names = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]

# Process only files in current batch
batch_cifs = cif_names[args.start:args.end]

power_spectrum_lmax_2 = spectra.Spectra(lmax=2, order=1, cutoff=cutoff)
power_spectrum_lmax_3 = spectra.Spectra(lmax=3, order=1, cutoff=cutoff)
power_spectrum_lmax_4 = spectra.Spectra(lmax=4, order=1, cutoff=cutoff)
power_spectrum_lmax_5 = spectra.Spectra(lmax=5, order=1, cutoff=cutoff)
power_spectrum_lmax_6 = spectra.Spectra(lmax=6, order=1, cutoff=cutoff)

bispectrum_lmax_2 = spectra.Spectra(lmax=2, order=2, cutoff=cutoff)
bispectrum_lmax_3 = spectra.Spectra(lmax=3, order=2, cutoff=cutoff)
bispectrum_lmax_4 = spectra.Spectra(lmax=4, order=2, cutoff=cutoff)
bispectrum_lmax_5 = spectra.Spectra(lmax=5, order=2, cutoff=cutoff)
bispectrum_lmax_6 = spectra.Spectra(lmax=6, order=2, cutoff=cutoff)

trispectrum_lmax_2 = spectra.Spectra(lmax=2, order=3, cutoff=cutoff)
trispectrum_lmax_3 = spectra.Spectra(lmax=3, order=3, cutoff=cutoff)
trispectrum_lmax_4 = spectra.Spectra(lmax=4, order=3, cutoff=cutoff)


for cif_file in tqdm(batch_cifs):
    print(f'Processing {cif_file}')
    cif_path = os.path.join(cif_dir, cif_file)
    power_spectrum_lmax_2.load_cif(cif_path)

    cif_name = cif_file.split('.')[0]
    structure = power_spectrum_lmax_2.get_structure()
    n_sites = len(structure)
    site_index = random.randint(0, n_sites - 1)

    try:
        local_geometry = power_spectrum_lmax_2.get_local_geometry(site_index)
    except:
        continue

    if local_geometry is None:
        continue

    sh_signal_lmax_2 = power_spectrum_lmax_2.compute_sh_signal(site_index)
    ps_lmax_2 = power_spectrum_lmax_2.compute_sh_signal_spectra(sh_signal_lmax_2)

    lgf = LocalGeometryFinder()
    lgf.setup_structure(structure)
    try:  # voronoi sometimes fails
        se = lgf.compute_structure_environments(
            structure,
            only_indices=[site_index]
        )
    except:
        continue

    lse = LightStructureEnvironments.from_structure_environments(
        strategy=strategy,
        structure_environments=se
    )
    coordination_environments = lse.coordination_environments[site_index]
    
    site_element = power_spectrum_lmax_2.get_site_element(site_index)
    local_elements = power_spectrum_lmax_2.get_local_elements(site_index)
    local_geometry = power_spectrum_lmax_2.get_local_geometry(site_index)

    bs_lmax_2 = bispectrum_lmax_2.compute_geometry_spectra(local_geometry)
    ts_lmax_2 = trispectrum_lmax_2.compute_geometry_spectra(local_geometry)

    sh_signal_lmax_3 = power_spectrum_lmax_3.compute_geometry_sh_signal(local_geometry)
    ps_lmax_3 = power_spectrum_lmax_3.compute_sh_signal_spectra(sh_signal_lmax_3)
    bs_lmax_3 = bispectrum_lmax_3.compute_sh_signal_spectra(sh_signal_lmax_3)  
    ts_lmax_3 = trispectrum_lmax_3.compute_sh_signal_spectra(sh_signal_lmax_3)

    sh_signal_lmax_4 = power_spectrum_lmax_4.compute_geometry_sh_signal(local_geometry)
    ps_lmax_4 = power_spectrum_lmax_4.compute_sh_signal_spectra(sh_signal_lmax_4)
    bs_lmax_4 = bispectrum_lmax_4.compute_sh_signal_spectra(sh_signal_lmax_4)
    ts_lmax_4 = trispectrum_lmax_4.compute_sh_signal_spectra(sh_signal_lmax_4)

    sh_signal_lmax_5 = power_spectrum_lmax_5.compute_geometry_sh_signal(local_geometry)
    ps_lmax_5 = power_spectrum_lmax_5.compute_sh_signal_spectra(sh_signal_lmax_5)
    bs_lmax_5 = bispectrum_lmax_5.compute_sh_signal_spectra(sh_signal_lmax_5)

    sh_signal_lmax_6 = power_spectrum_lmax_6.compute_geometry_sh_signal(local_geometry)
    ps_lmax_6 = power_spectrum_lmax_6.compute_sh_signal_spectra(sh_signal_lmax_6)
    bs_lmax_6 = bispectrum_lmax_6.compute_sh_signal_spectra(sh_signal_lmax_6)

    ts_lmax_4 = trispectrum_lmax_4.compute_geometry_spectra(local_geometry)
    
    label_vector = np.zeros(len(labels))
    for coordination_environment in coordination_environments:
        ce_symbol = coordination_environment['ce_symbol']
        ce_fraction = coordination_environment['ce_fraction']
        label_vector[labels.index(ce_symbol)] = ce_fraction
    top_label = labels[np.argmax(label_vector)]

    local_env_dict = {
        'cif_name': cif_name,
        'site_index': site_index,
        'site_element': site_element,
        'local_elements': local_elements,
        'local_geometry': local_geometry,
        'sh_signal_lmax_2': sh_signal_lmax_2.array,
        'sh_signal_lmax_3': sh_signal_lmax_3.array,
        'sh_signal_lmax_4': sh_signal_lmax_4.array,
        'sh_signal_lmax_5': sh_signal_lmax_5.array,
        'sh_signal_lmax_6': sh_signal_lmax_6.array,
        'power_spectrum_lmax_2': ps_lmax_2,
        'power_spectrum_lmax_3': ps_lmax_3,
        'power_spectrum_lmax_4': ps_lmax_4,
        'power_spectrum_lmax_5': ps_lmax_5,
        'power_spectrum_lmax_6': ps_lmax_6,
        'bispectrum_lmax_2': bs_lmax_2,
        'bispectrum_lmax_3': bs_lmax_3,
        'bispectrum_lmax_4': bs_lmax_4,
        'bispectrum_lmax_5': bs_lmax_5,
        'bispectrum_lmax_6': bs_lmax_6,
        'trispectrum_lmax_2': ts_lmax_2,
        'trispectrum_lmax_3': ts_lmax_3,
        'trispectrum_lmax_4': ts_lmax_4,
        'label_vector': label_vector,
        'top_label': top_label
    }
    df.loc[len(df)] = local_env_dict

# Save results for this batch
output_dir = os.path.join(project_root, 'data/mp/bispectra_labels/')
os.makedirs(output_dir, exist_ok=True)
df.to_csv(
    os.path.join(
        output_dir,
        f'mp_bispectra_labels_batch_{args.start}_{args.end}.csv'
    ),
    index=False
)