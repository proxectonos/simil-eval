#!/bin/bash

#SBATCH -D .             
#SBATCH -o ../logs_similarity/debug_refactor_%j_err.log  
#SBATCH -e ../logs_similarity/debug_refactor_%j_out.log
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --nodes 1
#SBATCH --mem=50G
#SBATCH -t 30:00

module --force purge
module load cesga/2020
module load python/3.9.9
source ../../eval_env/bin/activate

cd ..


MODEL="BSC-LT/salamandra-2b"
DATASET="openbookqa"
LANGUAGE="gl"
SHOW_OPTIONS="True"
FEWSHOT_NUM="5"
TOKEN_HF=""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""
CACHE_DIR="/mnt/netapp1/Proxecto_NOS/adestramentos/avaliacion/similaridade_framework/cache"

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
    --create_examples \
    --fewshot_num $FEWSHOT_NUM \
    --show_options $SHOW_OPTIONS \
    --examples_file $EXAMPLES_FILE \
    --generate_completions \
    --results_file $RESULTS_FILE \
    --evaluate_similarity \
    --metrics cosine moverscore bertscore