import warnings
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import src.spectra as spectra
from pymongo import MongoClient
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings("ignore")

lmax = 4
cif_dirs = ['mp/AgS']
cutoff = spectra.min_dist_cutoff()
analysis_name = "mp_AgS_bispectra_lmax_4_min_dist_cutoff"
# neighbors = ['S']
neighbors = []

cif_dirs = [os.path.join(os.getcwd(), f"../cifs/{cif_dir}") for cif_dir in cif_dirs]
database = sys.argv[1] # either "mongodb" or "csv"
base_dir = os.getcwd()
bispectrum = spectra.Spectra(lmax=4, order=2, cutoff=cutoff, neighbors=neighbors)

if database == "mongodb":
    client = MongoClient()
    db = client['spectra_database']
    collection = db[analysis_name]
    collection.delete_many({})
elif database == "csv":
    df = pd.DataFrame(columns=['cif_name', 'formula', 'site_index', 'site_element', 'local_geometry', 'local_elements', 'bispectrum'])

for cif_dir in cif_dirs:
    os.chdir(cif_dir)
    cif_names = os.listdir()
    for cif_name in tqdm(cif_names):
        cif_path = os.path.join(os.getcwd(), cif_name)
        bispectrum.load_cif(cif_path)
        # spectra = bispectrum.compute_structure_spectra(symmetry_unique_only=True)
        spectra = bispectrum.compute_element_spectra('Ag')
        for site_index, spectrum in spectra.items():
            if spectrum is not None:
                local_env_dict = {
                    'cif_name': cif_name,
                    'formula': bispectrum.get_formula(),
                    'site_index': site_index,
                    'site_element': bispectrum.get_site_element(site_index),
                    'local_geometry': bispectrum.get_local_geometry(site_index).tolist(),
                    'local_elements': bispectrum.get_local_elements(site_index),
                    'bispectrum': spectrum.tolist()
                }

                if database == "mongodb":
                    collection.insert_one(local_env_dict)

                elif database == "csv":
                    df.loc[len(df)] = local_env_dict

if database == "csv":
    os.chdir(base_dir)
    df.to_csv(f'{analysis_name}.csv', index=False)
    print(f"Saved {os.path.join(base_dir, f'{analysis_name}.csv')}")
