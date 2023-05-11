#!/bin/bash
#
## List of split_length
#all_split_length=(500 1000 1500 2000 2500)
#embedding_threshold=(95 90 85 80 75 70 65 60 55 50)
#overlap_lengths=(1 2 3 4 5 6 7 8 9 10 11)
#
#cd ../model-code
#
## Loop through each split_length
#for length in "${overlap_lengths[@]}"
#do
#    # Execute the python script and store the last line of the result in a variable
#    python hybrid.py --saved_suffix droc_c2f_May12_17-38-53_1800 --config_name droc_final --method overlapping --split_length 1500 --results_output ../results/logs --use_c2f --overlapping_length "$length"
#    #result=$(python hybrid.py --saved_suffix droc_incremental_no_segment_distance_May02_17-32-58_1800 --config_name droc_final_inc_512 --method neural --split_length "$split_length" --results_output ../results/logs --use_c2f | tail -1)
#
#
#
#done

# droc c2f
python hybrid.py --saved_suffix droc_c2f_May12_17-38-53_1800 --config_name droc_final_512 --method embedding --embedding_threshold 90 --embedding_method fastText --split_length 512 --results_output ../results/logs --use_c2f
python hybrid.py --saved_suffix droc_c2f_May12_17-38-53_1800 --config_name droc_final_512 --method embedding --embedding_threshold 90 --embedding_method word2vec --split_length 512 --results_output ../results/logs --use_c2f

# droc gold
python hybrid.py --saved_suffix droc_c2f_May12_17-38-53_1800 --config_name droc_final_512 --method embedding --embedding_threshold 90 --embedding_method fastText --split_length 512 --results_output ../results/logs
python hybrid.py --saved_suffix droc_c2f_May12_17-38-53_1800 --config_name droc_final_512 --method embedding --embedding_threshold 90 --embedding_method word2vec --split_length 512 --results_output ../results/logs

# news c2f
python hybrid.py --saved_suffix tuba10_gelectra_large_Apr20_15-37-33_54802 --config_name droc_final_news --method embedding --embedding_threshold 90 --embedding_method fastText --split_length 512 --results_output ../results/logs --use_c2f
python hybrid.py --saved_suffix tuba10_gelectra_large_Apr20_15-37-33_54802 --config_name droc_final_news --method embedding --embedding_threshold 90 --embedding_method word2vec --split_length 512 --results_output ../results/logs --use_c2f

# news gold
python hybrid.py --saved_suffix tuba10_gelectra_large_Apr20_15-37-33_54802 --config_name droc_final_news --method embedding --embedding_threshold 90 --embedding_method fastText --split_length 512 --results_output ../results/logs
python hybrid.py --saved_suffix tuba10_gelectra_large_Apr20_15-37-33_54802 --config_name droc_final_news --method embedding --embedding_threshold 90 --embedding_method word2vec --split_length 512 --results_output ../results/logs