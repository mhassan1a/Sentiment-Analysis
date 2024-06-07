#!/bin/bash
#SBATCH --partition=gpu       # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --tasks-per-node=32       # number of tasks per node
#SBATCH --time=0-24:00:00         # total runtime of job allocation
#SBATCH --gres=gpu:1             # number of general-purpose GPUs
#SBATCH --mem=50G               # memory per node in MB
#SBATCH --output=./out/train_net-%j.out    # filename for STDOUT
#SBATCH --error=./out/train_net-%j.err     # filename for STDERR

echo "Starting Job"

python ./src/train.py --dataset goemotions\
                    --batch_size 128\
                    --epochs 20\
                    --device cuda\
                    --output_name transformer_model.pt\
                    --output_path checkpoints\
                    --model_name transformer\
                    --n_heads 16\
                    --vocab_size 30522\
                    --emb_dim 768\
                    --hidden_dim 32\
                    --dropout 0.3\
                    --weight_decay 0.00005\
                    --n_layers 2\
                    --lr 0.0001\
                    --dry_run 0\
                    --seed 3423452\
                    --test 0\
                    --n_workers 4\
                    --job_id $SLURM_JOB_ID\
                    --dim_feedforward 2048\

# dataset: goemotions, yelp
# model_name: transformer, lstm