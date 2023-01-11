#!/bin/bash
#SBATCH --nodes 4            # Request 4 nodes 
#SBATCH --gres=gpu:2         # Request 2 GPU "generic resources” per node (max 12 per user).
#SBATCH --tasks-per-node=2   # Request 1 process per GPU. You will get 1 CPU per process by default.
#SBATCH --cpus-per-tasks=1   # Request 1 CPU per process. You may request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --mem=8G      
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out
#SBATCH --account=<your account> # Replace the template with your account name

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
echo "Installing requirements..."
python bin/install_requirements.py --no-index || exit 1

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script..."

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

srun python pytorch-ddp-test.py --init_method tcp://$MASTER_ADDR:3456 \
                                --world_size $SLURM_NTASKS \
                                --config_file "config.ini" \
                                --dist_backend "nccl" \
                                --num_workers $SLURM_CPUS_PER_TASK