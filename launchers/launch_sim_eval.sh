#!/bin/bash

#SBATCH -D .             
#SBATCH -o ../logs_similarity/%x_%j_err.log  
#SBATCH -e ../logs_similarity/%x_%j_out.log
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --nodes 1
#SBATCH --mem=70G
#SBATCH -t 2:00:00

module --force purge
module load cuda/12.8.0
module load python/3.10.8
module load cesga/2022
source ../eval_env/bin/activate

cd ..

MODEL=$1
DATASET=$2
LANGUAGE=$3
SHOW_OPTIONS=$4
FEWSHOT_NUM=$5
CACHE_DIR=$6
TOKEN_HF=$7

modelname=${MODEL##*/}
EXAMPLES_FILE="texts/${DATASET}_${LANGUAGE}_${FEWSHOT_NUM}fewshot_${SHOW_OPTIONS}options_examples.txt"
RESULTS_FILE="generated_files/${modelname}_${DATASET}_${LANGUAGE}_${FEWSHOT_NUM}fewshot_${SHOW_OPTIONS}options_${SLURM_JOB_ID}.csv"
        
echo "Launching evaluation of model $MODEL on the dataset $DATASET with $FEWSHOT_NUM fewshot examples ..."  
python3 eval_similarity.py \
    --dataset $DATASET \
    --cache $CACHE_DIR \
    --token $TOKEN_HF \
    --model $MODEL \
    --language $LANGUAGE \
    --evaluate_similarity \
    --create_examples \
    --fewshot_num $FEWSHOT_NUM \
    --show_options $SHOW_OPTIONS \
    --examples_file $EXAMPLES_FILE \
    --generate_completions \
    --results_file $RESULTS_FILE \
    --evaluate_similarity \
    --metrics cosine moverscore bertscore