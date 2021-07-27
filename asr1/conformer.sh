#!/bin/bash
#
#SBATCH --job-name="conformer-esp1"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --time="24:00:00"
#SBATCH --gres=gpu:4
#SBATCH --error=/home/mezzat/esp1-conformer.out
#SBATCH --output=/home/mezzat/esp1-conformer.out
#SBATCH --partition gpu

module purge

module load slurm/slurm/19.05.7
module load espnet/0.9.10-PyTorch-1.7.1/0.9.10-PyTorch-1.7.1
module load Python/3.8.6-GCCcore-10.2.0

srun ./run.sh --stage 1 --stop_stage 1
