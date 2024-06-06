#!/bin/bash
#SBATCH --partition=gpu       # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --tasks-per-node=32       # number of tasks per node
#SBATCH --time=0-24:00:00         # total runtime of job allocation
#SBATCH --gres=gpu:1             # number of general-purpose GPUs
#SBATCH --mem=150G               # memory per node in MB
#SBATCH --output=./out/train_net-%j.out    # filename for STDOUT
#SBATCH --error=./out/train_net-%j.err     # filename for STDERR

echo "Starting Job"

python src/train.py --dataset goemotions\
                    --batch_size 32\
                    --epochs 5\
                    --device cuda\
                    --output_name model.pt\
                    --output_path checkpoints\
                    --vocab_size 30522\
                    --emb_dim 300\
                    --hidden_dim 256\
                    --dropout 0.2\
                    --n_layers 2\
                    --lr 0.001\
                    --dry_run False\
                    --seed 42
