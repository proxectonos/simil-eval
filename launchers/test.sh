cd ..

python3 eval_similarity_v2.py \
    --dataset belebele \
    --cache /mnt/netapp1/Proxecto_NOS/adestramentos/avaliacion/cache \
    --language pt \
    --create_examples \
    --examples_file texts/belebele_pt_5fewshot_Trueoptions_examples.txt \
    --fewshot_num 5 \
    --show_options True