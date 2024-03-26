# This is a script to extract 50 random samples from the ICSD all with different space groups
# with the intention of seeing how different cutoff choices perform

import os
import shutil

count = 0
space_groups = set()
files_to_copy = []
for filename in os.listdir("../cifs/icsd_mini"):
    space_group = int(filename.split('_')[1][:-4]) # example filename: ZrRhGa_189.cif
    if space_group not in space_groups:
        space_groups.add(space_group)
        files_to_copy.append(filename)
        count += 1
    if count == 50:
        break

os.makedirs("../cifs/icsd_samples", exist_ok=True)

for filename in files_to_copy:
    shutil.copy(f"../cifs/icsd_mini/{filename}", f"../cifs/icsd_samples/{filename}")
    print(f"copied {filename}")