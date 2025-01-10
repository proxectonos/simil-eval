#!/bin/bash

#CONFIGURATION -----------------------------------------------------------------
MODELS=(
#"proxectonos/Carballo-cerebras-1.3B"
#"proxectonos/Carballo-bloom-1.3B"
#"Nos-PT/Carvalho_pt-gl-1.3B"
#"google/gemma-2-2b"
#"NOVA-vision-language/GlorIA-1.3B"
#"utter-project/EuroLLM-1.7B"
#"proxectonos/Llama-3.1-Carballo"
#"meta-llama/Llama-3.1-8B"
#"meta-llama/Llama-3.1-8B-Instruct"
#"BSC-LT/salamandra-2b"
#"BSC-LT/salamandra-7b"
#"BSC-LT/salamandraTA-2B"
#"irlab-udc/Llama-3.1-8B-Instruct-Galician"
#"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment1_10-09-24_21-29/checkpoint-2224"
#"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment2_10-16-24_12-20/checkpoint-8772"
#"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment3_10-09-24_21-29/checkpoint-2266"
#"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment4_10-25-24_14-47/checkpoint-9816"
#"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment5_12-06-24_20-00/checkpoint-2453"
#"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Experimento3_annealing_lr6_gr2_experiment6_12-12-24_21-11/checkpoint-244"
#"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Experimento3_annealing_lr6_2Epochs_experiment6_12-13-24_09-53/checkpoint-488"
#"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Experimento3_annealing_lr6_3Epochs_experiment6_12-18-24_02-52/checkpoint-1464"
"/mnt/netapp1/Proxecto_NOS/adestramentos/Carvalho-Llama/89_percent"
)

DATASETS=(
    "openbookqa"
    #"belebele"
)

LANGUAGES=(
    "gl"
    "cat"
    "en"
    "es"
    "pt"
)

SHOW_OPTIONS="True"
FEWSHOT_NUM=5
############################################################################################################

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
for dataset in "${DATASETS[@]}";do
    for language in "${LANGUAGES[@]}"; do
        for model in "${MODELS[@]}"; do
            modelname=${model##*/}
            #modelname="Experimento6-Annealing-3Epoch"
            job_name="similarity_${date}_${modelname}_${dataset}_${language}_${FEWSHOT_NUM}fewshot_${SHOW_OPTIONS}options"
            echo "Launching job $job_name"
            sbatch -J "$job_name" launch_sim_eval.sh $model $dataset $language $SHOW_OPTIONS $FEWSHOT_NUM $CACHE_DIR $HF_TOKEN
        done
    done
done

echo "Launched done"


