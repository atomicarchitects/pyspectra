#!/bin/bash

# Total number of files to process
TOTAL_FILES=181473

# Number of files per job
# FILES_PER_JOB=4537
FILES_PER_JOB=757


# Calculate how many jobs are needed
NUM_JOBS=$((TOTAL_FILES / FILES_PER_JOB))

for ((i=0; i<NUM_JOBS; i++)); do
    START=$((i * FILES_PER_JOB))
    END=$((START + FILES_PER_JOB))

    # Submit the job
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=icsd_analysis_$i
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=04:00:00
#SBATCH --mem=4G


python ./icsd.py csv --start $START --end $END
EOT

done