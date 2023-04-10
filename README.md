## Setup steps
The setup steps are more or less the same as described in [model-code/README.md](model-code/README.md)
Steps contain just some extra information on how to get the data and how to evaluate different splits of the document and different 
 - Create folder `/model-code/data`
 - Create file `local.conf` based on `local.conf.example` and update the `base_dir` setting
 - For Deutscher Romankorpus (DROC) clone [the git repository](https://gitlab2.informatik.uni-wuerzburg.de/kallimachos/DROC-Release) somewhere
 - Install Python3 dependencies: `pip install -r requirements.txt`
 - Install Pytorch from https://pytorch.org/get-started/locally/
 - All splits created using the `split_*` Python scripts will need to be processed using `preprocess.py` to be used as training input for the model, for example, to split the DROC dataset run:
   - MAC-OS: `python split_droc.py --type-system-xml /Users/jan/Documents/Studium/Bachelorarbeit/DROC-Release/droc/src/main/resources/CorefTypeSystem.xml /Users/jan/Documents/Studium/Bachelorarbeit/DROC-Release/droc/DROC-xmi data/droc_test/german.droc_gold_conll`
   - WIN: `python split_droc.py --type-system-xml /d/Bachelorarbeit/droc-release/droc/src/main/resources/CorefTypeSystem.xml /d/Bachelorarbeit/droc-release/droc/DROC-xmi data/test/german.droc_gold_conll`
 - Then run the following command to preprocess the data into the correct format
   - `python preprocess.py --input_dir data/droc_final_news --output_dir data/droc_final_news --seg_len 128 --language german --tokenizer_name german-nlp-group/electra-base-german-uncased --input_suffix tuba10_gold_conll --input_format conll-2012 --model_type electracd`

 - Download pretrained Word2Vec vectors from https://www.deepset.ai/german-word-embeddings