#!/bin/bash

# Submit jobs for different lmax and order combinations
for lmax in 4; do
    for order in 1 2 3; do
        for chunk in $(seq 0 0); do
            # Submit the job
            sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=recon_l${lmax}_o${order}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mem=16G

python ./reconstruction.py --lmax $lmax --order $order --chunk $chunk
EOT
        done
    done
done