#!/bin/bash
#SBATCH --nodes 1            # Request 4 nodes
#SBATCH --mem=8G
#SBATCH --time=0-24:00
#SBATCH --output=/home/scolombo/slurmout/compliance-%N-%j.out
#SBATCH --account=def-lplevass 
#SBATCH --mail-user=samuele.colombo1@studenti.unimi.it
#SBATCH --mail-type=ALL
#SBATCH --array=1-10%1

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
echo "Installing requirements..."
python /home/scolombo/transient_detection/bin/install_requirements.py @beluga --no-index || exit 1
# pip install -e $SLURM_TMPDIR/transient_detection || exit 1
pip install -e /home/scolombo/transient_detection|| exit 1
echo "requirements installed: "

srun python /home/scolombo/transient_detection/bin/check_compliance.py --config_file /home/scolombo/transient_detection/config.ini 
