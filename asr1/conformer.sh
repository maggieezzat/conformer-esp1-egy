#!/bin/bash
#
#SBATCH --job-name="all-conformer-esp1"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --time="24:00:00"
#SBATCH --gres=gpu:4
#SBATCH --error=/home/mezzat/all-esp1-conformer.out
#SBATCH --output=/home/mezzat/all-esp1-conformer.out
#SBATCH --partition gpu

module purge

module load slurm/slurm/19.05.7
module load matplotlib/3.2.0-fosscuda-2020b
module load espnet/0.9.10-PyTorch-1.7.1/0.9.10-PyTorch-1.7.1
module load Python/3.8.6-GCCcore-10.2.0
module load chainer/6.0.0-fosscuda-2020b
module load tensorboardX/2.1-fosscuda-2020b-PyTorch-1.7.1
module load cryptography/3.4.7-fosscuda-2020b

srun ./run.sh --stage 2 --stop_stage 2 --resume 'exp/train_combined_pytorch_train_specaug/results/snapshot.ep.14'
