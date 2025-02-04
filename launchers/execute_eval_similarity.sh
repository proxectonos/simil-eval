#!/bin/bash

#CONFIGURATION -----------------------------------------------------------------
MODELS=(
"proxectonos/Carballo-cerebras-1.3B"
"proxectonos/Carballo-bloom-1.3B"
"Nos-PT/Carvalho_pt-gl-1.3B"
"google/gemma-2-2b"
"NOVA-vision-language/GlorIA-1.3B"
"utter-project/EuroLLM-1.7B"
"proxectonos/Llama-3.1-Carballo"
"meta-llama/Llama-3.1-8B"
#"meta-llama/Llama-3.1-8B-Instruct"
# "BSC-LT/salamandra-2b"
# "BSC-LT/salamandra-7b"
# "BSC-LT/salamandraTA-2B"
# "irlab-udc/Llama-3.1-8B-Instruct-Galician"
"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment1_10-09-24_21-29/checkpoint-2224"
"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment2_10-16-24_12-20/checkpoint-8772"
"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment3_10-09-24_21-29/checkpoint-2266"
"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment4_10-25-24_14-47/checkpoint-9816"
"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment5_12-06-24_20-00/checkpoint-2453"
"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Experimento3_annealing_lr6_3Epochs_experiment6_12-18-24_02-52/checkpoint-1464"
#"Nos-PT/Llama-Carvalho-PT"
#"Nos-PT/Llama-Carvalho-PT_2Epochs"
#"Nos-PT/Llama-Carvalho_1Epoch"
)

DATASETS=(
    #"openbookqa"
    #"belebele"
    #"veritasqa_mc1"
    #"summarization"
    #"xstorycloze"
    "truthfulqa_mc1"
)

LANGUAGES=(
    "gl"
    #"cat"
    #"en"
    # "es"
    # "pt"
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

YAML_TASKS_FILE="../configs/tasks_ubication.yaml"
is_valid_combination() {
    local dataset=$1
    local language=$2
    # Extract only the section for specified dataset and check for language
    sed -n "/^${dataset}:/,/^[a-z]/{/^[a-z]/!p}" "$YAML_TASKS_FILE" | grep -q "^  - ${language}:"
    return $?
}

date=$(date '+%d-%m-%Y')
for dataset in "${DATASETS[@]}";do
    for language in "${LANGUAGES[@]}"; do
        if is_valid_combination "$dataset" "$language"; then
            for model in "${MODELS[@]}"; do
                modelname=${model##*/}
                #modelname="Experimento6-Annealing-3Epoch"
                job_name="similarity_${date}_${modelname}_${dataset}_${language}_${FEWSHOT_NUM}fewshot_${SHOW_OPTIONS}options"
                echo "Launching job $job_name"
                sbatch -J "$job_name" launch_sim_eval.sh $model $dataset $language $SHOW_OPTIONS $FEWSHOT_NUM $CACHE_DIR $HF_TOKEN
            done
                else
            echo "Skipping invalid combination: dataset=${dataset}, language=${language}"
        fi
    done
done

echo "Launched done"


