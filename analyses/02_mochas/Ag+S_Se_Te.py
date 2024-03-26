
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from src.spectra import Spectra, radial_cutoff
from pymatgen.core.structure import Structure
from pymongo import MongoClient
import e3nn_jax as e3nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import jax.numpy as jnp

rerun = len(sys.argv) > 1 and sys.argv[1] == 'rerun'
# cif_file_dirs = ["mp/AgS", "mp/AgSe", "mp/AgTe"]
cif_file_dirs = ["../cif_files/mp/AgS"]

bispectrum = Spectra(lmax=4, order=2, neighbors=["S", "Se", "Te"], cutoff=radial_cutoff(radius=3.0))

client = MongoClient()
db = client['spectra_database']
analysis_name = "AgS_bispectra_lmax4_radius3"
collection = db[analysis_name]
if rerun:
    collection.delete_many({})
    for cif_file_dir in cif_file_dirs:
        os.chdir(cif_file_dir)

        cif_file_names = os.listdir()

        for cif_file_name in tqdm(cif_file_names):

            cif_file_path = os.path.join(os.getcwd(), cif_file_name)
            bispectrum.load_cif(cif_file_path)
            
            try:
                silver_spectra = bispectrum.compute_element_spectra("Ag")    
                for silver_site_number, spectrum in silver_spectra.items():
                    if spectrum is not None:
                        spectrum = spectrum / jnp.linalg.norm(spectrum)
                        existing_doc = collection.find_one({'cif_filepath': cif_file_path, 'atom_site_number': silver_site_number})
                        if existing_doc is None:
                            collection.insert_one({'cif_filepath': cif_file_path, 'atom_site_number': silver_site_number, 'spectra': spectrum.tolist()})
            except Exception as e:
                print(cif_file_path, bispectrum)
                print("Exception occurred: ", e)
                

        os.chdir("..")

spectra_data = [doc['spectra'] for doc in collection.find()]
kmeans = KMeans(n_clusters=8, n_init=10, random_state=0).fit(spectra_data)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(spectra_data)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
colors = ['black' for _ in range(len(spectra_data))]
color_labels = [colors[label] for label in kmeans.labels_]

plt.figure(figsize=(10, 10))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=color_labels, s=5)
plt.title('PCA of the Bispectra', fontsize=20)
plt.xlabel('First Principal Component', fontsize=16)
plt.ylabel('Second Principal Component', fontsize=16)


print(os.getcwd())
os.chdir("../mochas")
cif_filenames = os.listdir()
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
markers = ['*', 'v', '^', '<', '>', 's', 'p', 'o']
legend_labels = set()
for i, cif_filename in enumerate(tqdm(cif_filenames)):
    cif_filepath = os.path.join(os.getcwd(), cif_filename)
    structure = Structure.from_file(cif_filepath)
    ag_sites = [i for i, site in enumerate(structure) if site.species_string == "Ag"]
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    for ag_site in ag_sites:
        bispectrum.load_cif(cif_filepath)
        spectra = bispectrum.compute_atom_spectra(ag_site)
        if spectra is not None:
            spectra = spectra / jnp.linalg.norm(spectra)
            pca_result = pca.transform([spectra])[0]
            print('cif_filepath:', cif_filepath, 'ag_site:', ag_site, 'pca_result:', pca_result)
            plt.scatter(pca_result[0], pca_result[1], marker=marker, s=500, alpha=0.5, color=color)  # Increased size from 200 to 500
            if color not in legend_labels:
                plt.scatter([], [], marker=marker, s=500, alpha=0.5, label=f"{cif_filename}", color=color)  # Increased size from 200 to 500
                legend_labels.add(color)
    break

# plt.legend(loc='upper right', framealpha=0.5)





# os.chdir("../cif_files/mochas")
# cif_filenames = os.listdir()
# colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
# markers = ['o', 'v', '^', '<', '>', 's', 'p', '*']
# legend_labels = set()
# for i, cif_filename in enumerate(tqdm(cif_filenames)):
#     cif_filepath = os.path.join(os.getcwd(), cif_filename)
#     structure = Structure.from_file(cif_filepath)
#     ag_sites = [i for i, site in enumerate(structure) if site.species_string == "Ag"]
#     color = colors[i % len(colors)]
#     marker = markers[i % len(markers)]
#     for ag_site in ag_sites:
#         spectra = bispectrum.compute_cif_file_atom(cif_filepath, atom_site_number=ag_site)
#         if spectra is not None:
#             spectra = spectra / jnp.linalg.norm(spectra)
#             pca_result = pca.transform([spectra])[0]
#             print('cif_filepath:', cif_filepath, 'ag_site:', ag_site, 'pca_result:', pca_result)
#             plt.scatter(pca_result[0], pca_result[1], marker=marker, s=200, alpha=0.5, color=color)
#             if color not in legend_labels:
#                 plt.scatter([], [], marker=marker, s=200, alpha=0.5, label=f"{cif_filename}", color=color)
#                 legend_labels.add(color)

# plt.legend(loc='upper right', framealpha=0.5)

while not os.getcwd().endswith('pyspectra'):
    os.chdir("..")
if not os.path.exists('figures'):
    os.makedirs('figures')
plt.savefig(f'./figures/{analysis_name}_PCA.png')  

