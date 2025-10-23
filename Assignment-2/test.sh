#!/bin/bash
#SBATCH --job-name=gpu_training
#SBATCH --partition=Standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=2g.45gb:1
#SBATCH --output=logs/gpu_job_%j.log
#SBATCH --time=0-6:00:00

curl --location-trusted -u 22b1029:f07b3c72a55d9598cc17fa115061476c "https://internet-sso.iitb.ac.in/login.php"

# python test.py --lambda 0.01 --tau 0.2 --loss_fn infoNCE --learning_rate 0.001 --test_epoch 1
# python test.py --lambda 0.1 --tau 0.2 --loss_fn infoNCE --learning_rate 0.001 --test_epoch 1
# python test.py --lambda 0.01 --tau 0.05 --loss_fn infoNCE --learning_rate 0.001 --test_epoch 1 --task_num 3
# python test.py --lambda 0.01 --tau 0.05 --loss_fn infoNCE --learning_rate 0.001 --test_epoch 1 --task_num 4
# python test.py --lambda 0.1 --tau 0.05 --loss_fn infoNCE --learning_rate 0.001 --test_epoch 1

# python test.py --lambda 0.01 --margin 0.2 --loss_fn triplet --learning_rate 0.001 --test_epoch 1 --task_num 3
python test.py --lambda 0.01 --margin 0.2 --loss_fn triplet --learning_rate 0.001 --test_epoch 1 --task_num 4
# python test.py --lambda 0.1 --margin 0.2 --loss_fn triplet --learning_rate 0.001 --test_epoch 1
# python test.py --lambda 0.01 --margin 0.5 --loss_fn triplet --learning_rate 0.001 --test_epoch 1
# python test.py --lambda 0.1 --margin 0.5 --loss_fn triplet --learning_rate 0.001 --test_epoch 1