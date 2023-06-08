# Evaluating Rule-Based and Neural Merging Strategies of Entity Clusters for Coreference Resolution

This repository contains the code for the Bachelor Thesis with the title "Evaluating Rule-Based and Neural Merging Strategies of Entity Clusters for Coreference Resolution".
The repository is split into three main parts/folders
 - **/model-code/**: This is directory contains the important code. It is based on the [work of Schröder, Fynn and Hatzel, Hans Ole and Biemann, Chris](https://github.com/uhh-lt/neural-coref/tree/konvens). Most of the code is unchanged. The in this thesis adapted incremental model is part of the [*model.py*](/model-code/model.py) file. The split-and-merging command is located in the file [*split_an_merge.py*](/model-code/split_an_merge.py)
 - **/results/**: Contains log outputs for the experiments that were run for the evaluation. Also, the code to generate the plots is part of this folder.
 - **/visualization/**: A small visualization tool based on React that I used in the beginning to see how the coarse-to-fine model predicts the entities. This is not really part of the thesis and was just used for development purposes.

You can find the final thesis [here](https://www.inf.uni-hamburg.de/en/inst/ab/lt/teaching/theses/completed-theses/2023-ba-maegdefrau.pdf).

## Abstract
Coreference resolution is an essential pre-processing step for many natural language processing tasks. In the past, there has been a shift from rule-based approaches to machine-learning-based approaches. One of these approaches was proposed by Schröder et al., which involved two end-to-end trained models. The first model, called the coarse-to-fine model, suffers from the problem of requiring an increasing amount of memory. Meanwhile, the second model, called the incremental model, only requires a constant amount of memory but performs worse than the coarse-to-fine model. 

Unfortunately, when predicting mentions and entities in long documents, there is currently no well-performing method that requires only a limited amount of memory. To address this problem, we propose in this thesis the idea of splitting documents into shorter splits, predicting them using the coarse-to-fine model, and merging the entities together afterward. For the merging step, we propose three rule-based methods, including string-matching based, overlapping-based, and one based on word embeddings. Additionally, we also introduce a neural method that adapts the incremental model.

The results of our thesis show that all the proposed methods underperform the given baseline. However, they all work with a limited amount of memory. Furthermore, we found that the rule-based methods perform better than the more complex adapted incremental model.


## Setup steps
The setup steps are more or less the same as described in [model-code/README.md](model-code/README.md).
Steps contain just some extra information on how to get the data and how to evaluate different splits of the document and different 
 - Create folder `/model-code/data`
 - Create file `local.conf` based on `local.conf.example` and update the `base_dir` setting
 - For Deutscher Romankorpus (DROC) clone [the git repository](https://gitlab2.informatik.uni-wuerzburg.de/kallimachos/DROC-Release) somewhere
 - Install Python3 dependencies: `pip install -r requirements.txt`
 - Install Pytorch from https://pytorch.org/get-started/locally/
 - If you want to use pre-trained models you can download them of [the releases page](https://github.com/uhh-lt/neural-coref/releases/tag/konvens) of the code by Schröder et. al.. Unpack the archives and copy the model weights into the data-folder of the experiment that you are currently running.
 - All splits (of datasets into the dev/train/test files) created using the `split_*` Python scripts will need to be processed using `preprocess.py` to be used as training input for the model, for example, to split the DROC dataset run:
   - `python split_droc.py --type-system-xml /path/to/droc/src/main/resources/CorefTypeSystem.xml /path/to/droc/DROC-xmi data/test/german.droc_gold_conll`
   - `python preprocess.py --input_dir data/droc_final_news --output_dir data/droc_final_news --seg_len 128 --language german --tokenizer_name german-nlp-group/electra-base-german-uncased --input_suffix tuba10_gold_conll --input_format conll-2012 --model_type electrac`

 - When using the merging method based on word embeddings download pretrained Word2Vec vectors from https://www.deepset.ai/german-word-embeddings

## Command Line Arguments

The following command line arguments are available for running the script:

* `--config_name` (required): Name of the experiment that should be run. From the experiments.conf.
* `--saved_suffix` (required): Suffix of the model that will be run.
* `--method` (required): Merging method that should be applied. Available choices are: `STRING_MATCHING`, `NEURAL`, `EMBEDDING`, `OVERLAPPING`.
* `--split_length`: Maximum length of a split (default: 512).
* `--use_c2f`: If splits should be predicted with coarse-to-fine model.
* `--results_output` (required): Path were the results will be stored.
* `--embedding_method`: Type of word embeddings that are used. Available choices are: `fastText`, `word2vec` (default: word2vec).
* `--embedding_threshold`: Required cosine-similarity to merged two entities (default: 95).
* `--overlapping_length`: Number of sentence that should overlap (default: 0).
* `--exclude_pronouns`: If pronouns should be considered in string-based merging method

Example command: `python split_an_merge.py --saved_suffix droc_c2f_May12_17-38-53_1800 --config_name droc_final_512 --method embedding --embedding_threshold 90 --embedding_method fastText --split_length 512 --results_output ../results/logs --use_c2f`
