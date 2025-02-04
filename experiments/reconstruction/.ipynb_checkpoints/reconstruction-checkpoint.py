import sys
import os
import pickle
import argparse
import jax
import jax.numpy as jnp
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
src_dir = os.path.join(project_root, 'src')
sys.path.append(src_dir)

import spectra as spectra
from utils.alignment import align_signals, find_best_random_quaternion

parser = argparse.ArgumentParser()
parser.add_argument('--lmax', type=int, required=True)
parser.add_argument('--order', type=int, required=True)
parser.add_argument('--chunk', type=int, required=True)
args = parser.parse_args()

with open('../../data/qm9_local_envs_10000.pkl', 'rb') as f:
    local_envs = pickle.load(f)

local_envs = local_envs[args.chunk*1000:(args.chunk+1)*1000]

key = jax.random.PRNGKey(0)

print(f"lmax = {args.lmax}, order = {args.order}")

# Initialize the spectrum
spectrum = spectra.Spectra(lmax=args.lmax, order=args.order)

total_error = 0
for local_env in tqdm(local_envs):
    # Convert the local environment to a jax array
    local_env = jnp.array(local_env)
    
    # Compute the signal of the local environment 
    local_env_signal = spectra.sum_of_diracs(local_env, args.lmax)
    
    # Compute the spectrum of the local environment
    local_env_spectrum = spectrum.compute_geometry_spectra(local_env)

    # Invert the spectrum to get the predicted geometry
    predicted_geometry = spectrum.invert(local_env_spectrum)

    # Compute the signal of the predicted geometry
    predicted_signal = spectra.sum_of_diracs(predicted_geometry, args.lmax)

    # Align the predicted signal with the local environment signal
    initial_quaternion = find_best_random_quaternion(
        key, predicted_signal, local_env_signal)
    _, error = align_signals(predicted_signal, local_env_signal, initial_quaternion)

    # Add to the total error
    total_error += error

reconstruction_error = total_error / len(local_envs)
print(f"Reconstruction error: {reconstruction_error}")