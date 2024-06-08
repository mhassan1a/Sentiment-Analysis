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
                    --optimizer adam\
                    --momentum 0.2\
                    --epochs 100\
                    --device cuda\
                    --output_name lstm_model.pt\
                    --output_path checkpoints\
                    --model_name lstm\
                    --n_heads 4\
                    --vocab_size 30522\
                    --emb_dim 64\
                    --hidden_dim 64\
                    --dropout 0.3\
                    --weight_decay 0.0001\
                    --n_layers 3\
                    --lr 0.0001\
                    --dry_run 0\
                    --seed 3423452\
                    --test 0\
                    --n_workers 4\
                    --job_id $SLURM_JOB_ID\
                    --dim_feedforward 2048\
                    --use_bert_embeddings 1\

# dataset: goemotions, yelp
# model_name: transformer, lstm