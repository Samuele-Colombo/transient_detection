#!/bin/bash
#SBATCH --nodes 1            # Request 4 nodes 
#SBATCH --gres=gpu:4         # Request 2 GPU "generic resources” per node (max 12 per user).
#SBATCH --tasks-per-node=4   # Request 1 process per GPU.
#SBATCH --cpus-per-task=1    # Request 1 CPU per process. You may request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --mem-per-cpu=32G      
#SBATCH --time=0-12:00
#SBATCH --output=<output file>
#SBATCH --account=<account name> # Replace the template with your account name
#SBATCH --mail-user=<mail@domain>
#SBATCH --mail-type=ALL
# #SBATCH --array=1-10%1

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
echo "Installing requirements..."
python bin/install_requirements.py @beluga --no-index || exit 1
pip install -e <path to source>/transient_detection|| exit 1
echo "requirements installed: "
pip list

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20

export NCCL_BLOCKING_WAIT=3600000  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script..."

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

srun python /home/scolombo/transient_detection/bin/main.py --distributed_init_method tcp://$MASTER_ADDR:3456 \
                        --world_size $SLURM_NTASKS \
                        --config_file "config.ini" \
                        --dist_backend "nccl" \
                        --num_workers $SLURM_CPUS_PER_TASK \
                        # --fast #uncomment if data is already processed
                        