import os
import pandas as pd
from tqdm import tqdm

# Path to the directory containing csv files
csv_dir = './csvs'

# List to hold data from each csv file
dataframes = []

# Loop through each file in the directory
for filename in tqdm(os.listdir(csv_dir)):
    if filename.endswith('.csv'):
        filename_prefix = '_'.join(filename.split('_')[:-1])
        # Construct the full file path
        file_path = os.path.join(csv_dir, filename)
        # Read the csv file and append to the list
        dataframes.append(pd.read_csv(file_path))

# Concatenate all dataframes into one
print("Concatenating dataframes...")
combined_csv = pd.concat(dataframes)

# Save the combined csv to a new file
print("Saving combined CSV...")
combined_csv.to_csv(f'{filename_prefix}.csv', index=False)
