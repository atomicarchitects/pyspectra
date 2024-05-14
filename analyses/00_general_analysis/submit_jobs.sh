#!/bin/bash

# Directory containing the files
FILE_DIR="../../cifs/mp_AgS"

# Calculate the total number of files in the specified directory
TOTAL_FILES=$(ls -1q $FILE_DIR | wc -l)

# Number of files per job (ceiling of TOTAL_FILES / 240)
FILES_PER_JOB=$(( (TOTAL_FILES + 239) / 240 ))

# Calculate how many jobs are needed
NUM_JOBS=$(( (TOTAL_FILES + FILES_PER_JOB - 1) / FILES_PER_JOB ))

for ((i=0; i<NUM_JOBS; i++)); do
    START=$((i * FILES_PER_JOB))
    END=$((START + FILES_PER_JOB))
    if [ $END -gt $TOTAL_FILES ]; then
        END=$TOTAL_FILES
    fi

    # Create a temporary script for the job
    JOB_SCRIPT=$(mktemp)

    cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --job-name=icsd_analysis_$i
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=04:00:00
#SBATCH --mem=4G

python ./analysis.py csv --start $START --end $END
EOT

    # Submit the job
    sbatch $JOB_SCRIPT

    # Clean up the temporary script
    rm $JOB_SCRIPT

done
