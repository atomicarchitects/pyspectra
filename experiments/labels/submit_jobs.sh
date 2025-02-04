#!/bin/bash

# Total number of files to process
TOTAL_FILES=11000
# TOTAL_FILES=10

# Number of files per job
FILES_PER_JOB=50
# FILES_PER_JOB=10

# Calculate how many jobs are needed
NUM_JOBS=$((TOTAL_FILES / FILES_PER_JOB))

for ((i=0; i<NUM_JOBS; i++)); do
    START=$((i * FILES_PER_JOB))
    END=$((START + FILES_PER_JOB))

    # Submit the job
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=mp_${i}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mem=16G

python spectra_and_labels.py --start $START --end $END
EOT

done