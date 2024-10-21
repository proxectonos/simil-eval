#!/bin/bash

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

#CONFIGURATION -----------------------------------------------------------------
MODELS=(
# "proxectonos/Carballo-cerebras-1.3B"
#"proxectonos/Carballo-bloom-1.3B"
# "Nos-PT/Carvalho_pt-gl-1.3B"
# "projecte-aina/FLOR-1.3B"
#"bigscience/bloom-1b1"
#"bigscience/bloom-1b7"
#"ai-forever/mGPT"
# "google/gemma-2b"
# "NOVA-vision-language/GlorIA-1.3B"
"/mnt/netapp1/Proxecto_NOS/adestramentos/llama_trainings/output/Llama_experiment1_10-09-24_21-29/checkpoint-2224"
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
    #"pt"
)

SHOW_OPTIONS="True"
FEWSHOT_NUM=0
############################################################################################################

date=$(date '+%d-%m-%Y')
for dataset in "${DATASETS[@]}";do
    for language in "${LANGUAGES[@]}"; do
        for model in "${MODELS[@]}"; do
            #modelname=${model##*/}
            modelname="Experimento1"
            job_name="similarity_${date}_${modelname}_${dataset}_${language}_${FEWSHOT_NUM}fewshot_${SHOW_OPTIONS}options"
            echo "Launching job $job_name"
            sbatch -J "$job_name" launch_task.sh $model $dataset $language $SHOW_OPTIONS $FEWSHOT_NUM $CACHE_DIR $HF_TOKEN
        done
    done
done

echo "Launched done"


