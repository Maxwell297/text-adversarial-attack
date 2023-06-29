#!/bin/bash
#SBATCH --job-name=transfer-attack
#SBATCH --output=transfer-attack.out.%j
#SBATCH --error=transfer-attack.out.%j
#SBATCH --time=48:00:00
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15gb
#SBATCH --gres=gpu:1

# python text_classification.py --data_folder=data --dataset=mnli --finetune=True > logs/text_classification.log
# python evaluate_adv_samples.py --data_folder=data --dataset=mnli --finetune=True --start_index=0 --num_samples=100 --end_index=100 --gumbel_samples=1000 > logs/transfer_attack.log
python mask_fill.py > logs/mask_fill_adv_loss_top-3.log