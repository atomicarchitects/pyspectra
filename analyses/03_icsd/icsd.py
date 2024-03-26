
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import src.spectra as spectra
from pymongo import MongoClient
from tqdm import tqdm
import pandas as pd

lmax = 4
cutoff = spectra.chemenv_cutoff(strategy='multi_weights')
cif_dirs = ['icsd']
analysis_name = "icsd_bispectra_lmax_4_chemenv_cutoff"

cif_dirs = [os.path.join(os.getcwd(), f"../cifs/{cif_dir}") for cif_dir in cif_dirs]
database = sys.argv[1] # either "mongodb" or "csv"
bispectrum = spectra.Spectra(lmax=4, order=2, cutoff=cutoff)

if database == "mongodb":
    client = MongoClient()
    db = client['spectra_database']
    collection = db[analysis_name]
    collection.delete_many({})
elif database == "csv":
    df = pd.DataFrame(columns=['cif_name', 'site_index', 'site_element', 'local_geometry', 'local_elements', 'bispectrum'])

i = 0
for cif_dir in cif_dirs:
    os.chdir(cif_dir)
    print('here')
    cif_names = os.listdir()
    for cif_name in tqdm(cif_names):
        print(cif_name)
        i += 1
        if i >= 20:
            break
        cif_path = os.path.join(os.getcwd(), cif_name)
        bispectrum.load_cif(cif_path)
        spectra = bispectrum.compute_structure_spectra(symmetry_unique_only=True)
        if spectra is not None:
            for site_index, spectrum in spectra.items():
                if spectrum is not None:

                    site_element = bispectrum.get_structure()[site_index].species.elements[0].element.symbol
                    local_elements = bispectrum.get_local_elements(site_index)
                    local_geometry = bispectrum.get_local_geometry(site_index)
                    local_env_dict = {
                        'cif_name': cif_name, 
                        'site_index': site_index,
                        'site_element': site_element,
                        'local_geometry': local_geometry,
                        'local_elements': local_elements,
                        'bispectrum': spectrum.tolist()
                    }
                    
                    if database == "mongodb":
                        collection.insert_one(local_env_dict)

                    elif database == "csv":
                        df.loc[len(df)] = local_env_dict

if database == "csv":
    df.to_csv(f'{analysis_name}.csv', index=False)
