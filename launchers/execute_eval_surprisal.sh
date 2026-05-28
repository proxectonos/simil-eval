#!/bin/bash

#CONFIGURATION -----------------------------------------------------------------
MODELS=(
"langtech-languagemodeling/salamandra-7b-dev"
"proxectonos/Llama-3.1-Carballo-Instr3"
"Nos-PT/Llama-Carvalho-PT-GL"
"proxectonos/Llama-3.1-Carballo"
"nos-dev/Legal-Carballo-SalamandraInstruct_v1"
"meta-llama/Llama-3.1-8B-Instruct"
"BSC-LT/salamandra-7b-instruct"
)

# Load the environment variables from the .env file
if [ -f ../configs/.env ]; then
    export $(cat ../configs/.env | xargs)
fi

if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN is not set. Please set it in the .env file, or maybe some models or datasets couldn't work correctly"
fi

if [ -z "$CACHE_DIR" ]; then
    echo "CACHE_DIR is not set. Please set it in the .env file"
    exit
fi

date=$(date '+%d-%m-%Y')

for model in "${MODELS[@]}"; do
    modelname=${model##*/}
    job_name="surprisal_${date}_${modelname}"
    echo "Launching job $job_name"
    sbatch -J "$job_name" launch_sur_eval.sh $model $CACHE_DIR $HF_TOKEN
done

echo "Launched done"


