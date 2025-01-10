#!/bin/bash

#SBATCH -D .
#SBATCH -N 1
#SBATCH -J NOS-surprisal-CoLA-Carvalho-Llama
#SBATCH --mem=70G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH -o ../logs_surprisal/Debug_CoLA_%j.out
#SBATCH -e ../logs_surprisal/Debug_CoLA_%j.err
#SBATCH -t 30:00

declare -a MODELS=(
    # "proxectonos/Carballo-cerebras-1.3B" 
    # "proxectonos/Carballo-bloom-1.3B" 
    # "Nos-PT/Carvalho_pt-gl-1.3B" 
    # "projecte-aina/FLOR-1.3B" 
    # "bigscience/bloom-1b1"
    # "bigscience/bloom-1b7"
    # "ai-forever/mGPT"
    #"google/gemma-2-2b" 
    #"utter-project/EuroLLM-1.7B"
    #"NOVA-vision-language/GlorIA-1.3B"
    # "cerebras/Cerebras-GPT-1.3B"
    # "EleutherAI/gpt-neo-1.3B"
    #"proxectonos/Llama3.1-Carballo"
    #"meta-llama/Llama-3.1-8B"
    "BSC-LT/salamandra-2b"
    #"BSC-LT/salamandra-7b"
    #"BSC-LT/salamandraTA-2B"
    #"proxectonos/Carballo_Llama_Test"
    #"catallama/CataLlama-v0.1-Base"
    #"irlab-udc/Llama-3.1-8B-Instruct-Galician"
    #"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment1_10-09-24_21-29/checkpoint-2224"
    #/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment2_10-16-24_12-20/checkpoint-8772
    #"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment3_10-09-24_21-29/checkpoint-2266"
    #"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment4_10-25-24_14-47/checkpoint-9816"
    #"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment5_12-06-24_20-00/checkpoint-2453"
    #"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/CarballoLlama_annealing_experiment6_12-09-24_18-18/checkpoint-122"
    #"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Experimento3_annealing_lr6_gr2_experiment6_12-12-24_21-11/checkpoint-244"
    #"meta-llama/Llama-3.1-8B-Instruct"
    #"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Experimento3_annealing_lr6_2Epochs_experiment6_12-13-24_09-53/checkpoint-488"
    #"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Experimento3_annealing_lr6_3Epochs_experiment6_12-18-24_02-52/checkpoint-1464"
    #"/mnt/netapp1/Proxecto_NOS/adestramentos/Carvalho-Llama"
    #"/mnt/netapp1/Proxecto_NOS/adestramentos/Carvalho-Llama/89_percent"
)

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

echo "Launching surprisal test for Galician------------------"
for model in "${MODELS[@]}"; do
    python3 eval_surprisal.py --model $model --cache $CACHE_DIR --dataset cola --lang gl --token $HF_TOKEN
done

echo "Launching surprisal test for English------------------"
for model in "${MODELS[@]}"; do
    python3 eval_surprisal.py --model $model --cache $CACHE_DIR --dataset cola --lang en --token $HF_TOKEN
done

echo "Launching surprisal test for Catalan------------------"
for model in "${MODELS[@]}"; do
    python3 eval_surprisal.py --model $model --cache $CACHE_DIR --dataset cola --lang cat --token $HF_TOKEN
done

echo "Launching surprisal test for Spanish------------------"
for model in "${MODELS[@]}"; do
    python3 eval_surprisal.py --model $model --cache $CACHE_DIR --dataset cola --lang es --token $HF_TOKEN
done