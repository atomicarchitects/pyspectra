import sys
import os
import argparse
from tqdm import tqdm
import pandas as pd

project_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
src_dir = os.path.join(project_root, 'src')
sys.path.append(src_dir)

import spectra as spectra
from utils.cutoffs import chemenv_cutoff

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)
parser.add_argument('--lmax', type=int, required=True)
args = parser.parse_args()


cutoff = chemenv_cutoff(strategy='multi_weights')
bispectrum = spectra.Spectra(lmax=args.lmax, order=2, cutoff=cutoff)

df = pd.DataFrame(
    columns=['cif_name', 'site_index', 'site_element', 'local_elements', 'bispectrum', 'bader_charge']
)

cif_dir = os.path.join(project_root, "data/aflow/cifs")
print('Processing directory:', cif_dir)
cif_names = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]

# Process only files in current batch
batch_cifs = cif_names[args.start:args.end]

bader_charges = pd.read_csv(os.path.join(project_root, 'data/aflow/aflow_bader_charges.csv'))

for cif_file in tqdm(batch_cifs):
    print(f'Processing {cif_file}')
    cif_path = os.path.join(cif_dir, cif_file)
    bispectrum.load_cif(cif_path)

    cif_name = cif_file.split('.')[0]

    structure_bader_charges_row = bader_charges[bader_charges['cif_name'] == cif_name]
    try:
        structure_bader_charges = structure_bader_charges_row.drop('cif_name', axis=1).values[0]
    except IndexError:
        continue

    try:
        bispectra = bispectrum.compute_structure_spectra(symmetry_unique_only=True)
    except:
        continue
    
    if bispectra is not None:
        for site_index, spectrum in bispectra.items():
            if spectrum is not None:
                site_element = bispectrum.get_site_element(site_index)
                local_elements = bispectrum.get_local_elements(site_index)
                bader_charge = structure_bader_charges[site_index]
                local_env_dict = {
                    'cif_name': cif_name,
                    'site_index': site_index,
                    'site_element': site_element,
                    'local_elements': local_elements,
                    'bispectrum': spectrum.tolist(),
                    'bader_charge': bader_charge
                }
                df.loc[len(df)] = local_env_dict

# Save results for this batch
output_dir = os.path.join(project_root, 'data/aflow/bispectra_bader_charges')
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, f'aflow_bispectra_lmax_{args.lmax}_batch_{args.start}_{args.end}.csv'), index=False)