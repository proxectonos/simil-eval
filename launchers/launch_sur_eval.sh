#!/bin/bash

#SBATCH -D .  
#SBATCH -N 1   
#SBATCH -e ../logs_surprisal/%x_%j_err.log        
#SBATCH -o ../logs_surprisal/%x_%j_out.log  
#SBATCH --mem=70G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH -t 45:00

module --force purge
module load cesga/2022
module load python/3.10.8
module load cuda/12.8.0
source ../eval_env/bin/activate

cd ..

MODEL=$1
CACHE_DIR=$2
TOKEN_HF=$3

#------------------------------------------------------------
echo "Launching surprisal Cola-test for Galician------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset cola --lang gl --token $HF_TOKEN

echo "Launching surprisal Calame-test for Galician------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset calame --lang gl --token $HF_TOKEN

echo "Launching surprisal GlobalPIQA-test for Galician------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset globalpiqa --lang gl --token $HF_TOKEN

echo "Launching surprisal Cola-test for English------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset cola --lang en --token $HF_TOKEN

echo "Launching surprisal GlobalPIQA-test for English------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset globalpiqa --lang en --token $HF_TOKEN

echo "Launching surprisal Cola-test for Catalan------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset cola --lang cat --token $HF_TOKEN

echo "Launching surprisal GlobalPIQA-test for Catalan------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset globalpiqa --lang cat --token $HF_TOKEN

echo "Launching surprisal Cola-test for Spanish------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset cola --lang es --token $HF_TOKEN

echo "Launching surprisal GlobalPIQA-test for Spanish------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset globalpiqa --lang es --token $HF_TOKEN

echo "Launching surprisal Calame-test for Portuguese------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset calame --lang pt --token $HF_TOKEN

echo "Launching surprisal GlobalPIQA-test for Portuguese------------------"
python3 eval_surprisal.py --model $MODEL --cache $CACHE_DIR --dataset globalpiqa --lang pt --token $HF_TOKEN
