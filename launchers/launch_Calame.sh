#!/bin/bash

#SBATCH -D .
#SBATCH -N 1
#SBATCH -J NOS-Experimento1-Calame
#SBATCH --mem=70G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH -o ../logs_surprisal/Experimento1_Minicons_Calame-PT_%j.out
#SBATCH -e ../logs_surprisal/Experimento1_Minicons_Calame-PT_%j.err
#SBATCH -t 1:00:00

cd ..

# Load the environment variables from the .env file
if [ -f ./configs/.env ]; then
    export $(cat ./configs/.env | xargs)
fi

if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN is not set. Please set it in the .env file, or maybe some models or datasets couldn't work correctly"
fi

if [ -z "$CACHE_DIR" ]; then
    echo "CACHE_DIR is not set. Please set it in the .env file"
    exit
fi
#------------------------------------------------------------
declare -a MODELS=(
    # "proxectonos/Carballo-cerebras-1.3B" 
    # "proxectonos/Carballo-bloom-1.3B" 
    # "Nos-PT/Carvalho_pt-gl-1.3B" 
    # "projecte-aina/FLOR-1.3B" 
    # "bigscience/bloom-1b1"
    # "bigscience/bloom-1b7"
    # "ai-forever/mGPT"
    # "google/gemma-2b" 
    # "NOVA-vision-language/GlorIA-1.3B"
    # "cerebras/Cerebras-GPT-1.3B"
    # "EleutherAI/gpt-neo-1.3B"
    #"proxectonos/Llama3.1-Carballo"
    #"meta-llama/Llama-3.1-8B"
    #"BSC-LT/salamandra-2b"
    #"BSC-LT/salamandra-7b"
    #"utter-project/EuroLLM-1.7B"
    #"proxectonos/Carballo_Llama_Test"
    #"irlab-udc/Llama-3.1-8B-Instruct-Galician"
    "/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment1_10-09-24_21-29/checkpoint-2224"
)

echo "Launching minicons test for Calame-PT------------------"
for model in "${MODELS[@]}"; do
    python3 eval_minicons.py \
        --model $model \
        --cache $CACHE_DIR \
        --token $HF_TOKEN \
        --dataset calame \
        --lang pt
done