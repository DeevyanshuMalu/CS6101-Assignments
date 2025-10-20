#!/bin/bash
#SBATCH --job-name=gpu_training
#SBATCH --partition=Standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=2g.45gb:1
#SBATCH --output=logs/gpu_job_%j.log
#SBATCH --time=0-6:00:00

curl --location-trusted -u 22b1029:f07b3c72a55d9598cc17fa115061476c "https://internet-sso.iitb.ac.in/login.php"

# python train.py --lambda 1 --loss_fn triplet
python train.py --lambda 0.1 --loss_fn triplet
# python train.py --lambda 0.01 --loss_fn triplet
# python train.py --lambda 0.001 --loss_fn triplet
# python train.py --lambda 1 --loss_fn infoNCE --tau 1
# python train.py --lambda 0.1 --loss_fn infoNCE --tau 1
# python train.py --lambda 0.01 --loss_fn infoNCE --tau 1
# python train.py --lambda 0.001 --loss_fn infoNCE --tau 1
# python train.py --lambda 1 --loss_fn infoNCE --tau 0.5
# python train.py --lambda 0.1 --loss_fn infoNCE --tau 0.5
# python train.py --lambda 0.01 --loss_fn infoNCE --tau 0.5
# python train.py --lambda 0.001 --loss_fn infoNCE --tau 0.5