import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.append(src_dir)
import spectra as spectra
from utils.cutoffs import crystalnn_cutoff

parser = argparse.ArgumentParser()
parser.add_argument('format', type=str, help='Output format')
parser.add_argument('--start', type=int, default=0, help='Start index')
parser.add_argument('--end', type=int, default=None, help='End index')
parser.add_argument('--file-dir', type=str, default=None, help='File directory')
args = parser.parse_args()

lmax = 4
cif_dir = os.path.join(os.getcwd(), args.file_dir)
analysis_name = f"{args.file_dir.split('/')[-1]}_bispectra_lmax_{lmax}"
site_element = 'Ag'
inclusive_neighbors = []
bispectrum = spectra.Spectra(lmax=lmax, order=2, cutoff=crystalnn_cutoff())
df_columns = ['cif', 'formula', 'index', 'symbol', 'geometry', 'neighbors', 'signal', 'bispectrum']
df = pd.DataFrame(columns=df_columns)

cif_files = sorted(os.listdir(cif_dir))[args.start:args.end]
for cif_file in tqdm(cif_files):
    print(f"Processing {cif_file}")
    cif_path = os.path.join(cif_dir, cif_file)

    try:
        bispectrum.load_cif(cif_path)
    except Exception as e:
        print(f"Error loading {cif_file}: {e}")
        continue

    if site_element:
        bispectra = bispectrum.compute_element_spectra(site_element, inclusive_neighbors=inclusive_neighbors)
    else:
        bispectra = bispectrum.compute_structure_spectra(symmetry_unique_only=True)
        
    for index, spectrum in bispectra.items():

        formula = bispectrum.get_formula()
        symbol = bispectrum.get_site_element(index)
        geometry = bispectrum.get_local_geometry(index)
        neighbors = bispectrum.get_local_neighbors(index)
        signal = spectra.with_peaks_at(geometry, lmax=lmax).array
        spectrum /= np.linalg.norm(spectrum)

        local_env_data = {
            'cif': cif_file,
            'formula': formula,
            'index': index,
            'symbol': symbol,
            'geometry': geometry,
            'neighbors': neighbors,
            'signal': signal,
            'bispectrum': spectrum.tolist()
        }
        
        df = pd.concat([df, pd.DataFrame([local_env_data])], ignore_index=True)



output_dir = os.path.join(project_root, 'outputs', analysis_name)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'{analysis_name}_start_{args.start}_end_{args.end}.csv')
df.to_csv(output_path, index=False)

