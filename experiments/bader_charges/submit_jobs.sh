#!/bin/bash

# Total number of files to process
TOTAL_FILES=56827

# Number of files per job
FILES_PER_JOB=500

# Calculate how many jobs are needed
NUM_JOBS=$((TOTAL_FILES / FILES_PER_JOB))

for lmax in 4 5; do
    for ((i=0; i<NUM_JOBS; i++)); do
        START=$((i * FILES_PER_JOB))
        END=$((START + FILES_PER_JOB))

        # Submit the job
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=bc_${lmax}_${i}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mem=16G

python compute_bispectra_bader_charges.py --start $START --end $END --lmax $lmax
EOT

    done
done