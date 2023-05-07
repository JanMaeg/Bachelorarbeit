#!/bin/bash

# List of split_length
all_split_length=(500 1000 1500 2000 2500)
embedding_threshold=(95 90 85 80 75 70 65 60 55 50)

cd ../model-code

# Loop through each split_length
for threshold in "${embedding_threshold[@]}"
do
    # Execute the python script and store the last line of the result in a variable
    python hybrid.py --saved_suffix tuba10_gelectra_large_Apr20_15-37-33_54802 --config_name droc_final_news --method embedding --embedding_method word2vec --split_length 500 --results_output ../results/logs --embedding_threshold "$threshold"
    python hybrid.py --saved_suffix tuba10_gelectra_large_Apr20_15-37-33_54802 --config_name droc_final_news --method embedding --embedding_method fastText --split_length 500 --results_output ../results/logs --embedding_threshold "$threshold"
    #result=$(python hybrid.py --saved_suffix droc_incremental_no_segment_distance_May02_17-32-58_1800 --config_name droc_final_inc_512 --method neural --split_length "$split_length" --results_output ../results/logs --use_c2f | tail -1)

done