#!/bin/bash

#CONFIGURATION -----------------------------------------------------------------
MODELS=(
#  "/mnt/netapp1/Proxecto_NOS/checkpoints_RES/CPT_Carvalho-SalamandraInstruct_LR_1e4_10-28-25_18-55"
#  "/mnt/netapp1/Proxecto_NOS/checkpoints_RES/CPT_Carvalho-SalamandraInstruct_LR_1e5_10-28-25_18-56"
#  "/mnt/netapp1/Proxecto_NOS/checkpoints_RES/CPT_Carvalho-SalamandraInstruct_LR_1e6_10-29-25_03-08"
#"/mnt/netapp1/Proxecto_NOS/checkpoints_RES/CPT_Carvalho-SalamandraInstruct_LR_2e5"
#"/mnt/netapp1/Proxecto_NOS/checkpoints_RES/CPT_Carvalho-SalamandraInstruct_LR_2e6_11-06-25_09-44"
#"langtech-languagemodeling/salamandra-7b-dev"
"/mnt/netapp1/Proxecto_NOS/checkpoints_RES/CPT_Carvalho-SalamandraInstruct_v1_11-12-25_11-49"
# "proxectonos/Llama-3.1-Carballo-Instr3"
# "Nos-PT/Llama-Carvalho-PT-GL"
# "proxectonos/Llama-3.1-Carballo"
# "nos-dev/Legal-Carballo-SalamandraInstruct_v1"
# "meta-llama/Llama-3.1-8B-Instruct"
# "BSC-LT/salamandra-7b-instruct"
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


