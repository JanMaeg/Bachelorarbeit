import operator
import os
import pickle

from run import Runner
import sys
import util
from tensorize import Tensorizer
from os.path import join
import json
import itertools
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)


def convert_to_torch_tensor(input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                            is_training, gold_starts, gold_ends, gold_mention_cluster_map):
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
    sentence_len = torch.tensor(sentence_len, dtype=torch.long)
    genre = torch.tensor(genre, dtype=torch.long)
    sentence_map = torch.tensor(sentence_map, dtype=torch.long)
    is_training = torch.tensor(is_training, dtype=torch.bool)
    gold_starts = torch.tensor(gold_starts, dtype=torch.long)
    gold_ends = torch.tensor(gold_ends, dtype=torch.long)
    gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long)
    return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
           is_training, gold_starts, gold_ends, gold_mention_cluster_map,


def evaluate(config_name, gpu_id, saved_suffix, out_file):
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(saved_suffix)
    model.eval_only = True

    config = util.initialize_config(config_name, create_dirs=False)

    # Eingabe: Ein Dokument mit Länge N
    # Ich splitte das Dokumente N in Z Teile
    # Das Modell evaluiert diese Z Teile
    # Ich füge die Cluster dieser Z Teile zusammen

    tensorizer = Tensorizer({
        'long_doc_strategy': "split",
        'bert_tokenizer_name': 'german-nlp-group/electra-base-german-uncased',
        'genres': ["n"],
        'max_num_speakers': 20,
        'max_training_sentences': 40,
        'max_segment_len': 128
    })
    language = config['language']
    max_seg_len = config['max_segment_len']
    path = join(config['data_dir'], f'test.{language}.{max_seg_len}.jsonlines')

    exclude_merge_tokens = True

    with open(path, 'r') as f:
        samples = [json.loads(line) for line in f.readlines()]

        tensor_samples = itertools.chain(
            *(tensorizer.tensorize_example(sample, False, False) for sample in samples)
        )
        tensor_samples = list(tensor_samples)

        example = [(doc_key, convert_to_torch_tensor(*tensor)) for doc_key, tensor in
                   tensor_samples]
        stored_info = tensorizer.stored_info

        cache_path = join(config['data_dir'], f'predictions.{language}.{max_seg_len}')

        if os.path.exists(cache_path):
            # Load cached tensors if exists
            with open(cache_path, 'rb') as f:
                predictions = pickle.load(f)
                logger.info('Loaded predictions from cache')
        else:
            predictions = runner.evaluate(
                model,
                example,
                stored_info,
                0,
                official=False,
                conll_path=runner.config['conll_test_path'],
                out_file=out_file,
                hybrid=True
            )

            with open(cache_path, 'wb') as f:
                pickle.dump(predictions, f)

        counter = 0
        offset = 0
        output = {}
        merged_clusters = []
        merged_cluster_indices = []
        # Iterate through all splits of the document
        for doc_key, cluster in predictions.items():
            token_index_to_cluster = {}
            token_index_to_cluster_merged = {}
            cluster_to_tokens = []
            combined_predictions = []

            for cluster_index, single_cluster in enumerate(cluster):

                corrected_single_cluster = []
                cluster_tokes = []

                for span in single_cluster:
                    start_token_index = stored_info['subtoken_maps'][doc_key][span[0]]
                    end_token_index = stored_info['subtoken_maps'][doc_key][span[1]]

                    for index in range(start_token_index, end_token_index + 1):
                        token_index_to_cluster[index] = cluster_index

                    tokens = stored_info['tokens'][doc_key][start_token_index: end_token_index + 1]
                    cluster_tokes.append(' '.join(tokens))

                    corrected_single_cluster.append(span)

                combined_predictions.append(corrected_single_cluster)
                cluster_to_tokens.append(np.array(cluster_tokes))

            offset += torch.sum(example[counter][1][3]).item()
            counter += 1

            # merge the clusters
            # for the first document we take all clusters as they are, no need to merge them with anything
            if counter == 1:
                merged_clusters = cluster_to_tokens

                for cluster_index, single_cluster in enumerate(combined_predictions):
                    for span in single_cluster:
                        start_token_index = stored_info['subtoken_maps'][doc_key][span[0]]
                        end_token_index = stored_info['subtoken_maps'][doc_key][span[1]]

                        for index in range(start_token_index, end_token_index + 1):
                            token_index_to_cluster_merged[index] = cluster_index
            else:
                # Für jeden Token aus dem Cluster gucken wir wie oft er in allen anderen Clustern vorkommt
                # Wenn es Token gibt, welche in nur einem weitern Cluster vorkommen
                # gucken wir welches Cluster dort am häufigsten ist
                # Diese beiden Cluster werden gemerged
                for cluster_index, cluster in enumerate(cluster_to_tokens):
                    flat_cluster = cluster.flatten()

                    result = {}
                    possible_merge_cluster = -1
                    matches_in_possible_cluster = 0
                    count_possible_merges = 0
                    merge_token = ''

                    for token in flat_cluster:
                        best_cluster_match = -1
                        max_count = -1
                        cluster_count_with_match = 0
                        if token in ["er", "sie", "sich", "sein", "seinem", "ihre", "ihr", "Sie", "Ich", "ich", "ihm", "ihn", "seine", "ihres", "seinen", "ihrer", "ihrem", "seiner", "ihren"] and exclude_merge_tokens:
                            result[token] = (best_cluster_match, max_count, cluster_count_with_match)
                            continue

                        for index, existing_cluster in enumerate(merged_clusters):
                            flat_existing_cluster = existing_cluster

                            count = np.count_nonzero(flat_existing_cluster == token)

                            if count > 0: cluster_count_with_match += 1

                            if count > 0 and max_count < count:
                                max_count = count
                                best_cluster_match = index

                        result[token] = (best_cluster_match, max_count, cluster_count_with_match)
                        if cluster_count_with_match == 1 \
                                and matches_in_possible_cluster == max_count \
                                and best_cluster_match != possible_merge_cluster:
                            count_possible_merges += 1

                        if cluster_count_with_match == 1 and matches_in_possible_cluster < max_count:
                            matches_in_possible_cluster = max_count
                            possible_merge_cluster = best_cluster_match
                            merge_token = token
                            count_possible_merges = 1

                    if count_possible_merges == 1:
                        print("Cluster wrude durch Token gemerged: " + merge_token)
                        merged_clusters[possible_merge_cluster] = np.concatenate(
                            (merged_clusters[possible_merge_cluster], cluster))

                        for span in combined_predictions[cluster_index]:
                            start_token_index = stored_info['subtoken_maps'][doc_key][span[0]]
                            end_token_index = stored_info['subtoken_maps'][doc_key][span[1]]

                            for index in range(start_token_index, end_token_index + 1):
                                token_index_to_cluster_merged[index] = possible_merge_cluster

                    elif count_possible_merges == 0:
                        print("Es wurde kein anderes Cluster gefunden, welches mit den Token übereinstimmt")
                        merged_clusters.append(cluster)
                        for span in combined_predictions[cluster_index]:
                            start_token_index = stored_info['subtoken_maps'][doc_key][span[0]]
                            end_token_index = stored_info['subtoken_maps'][doc_key][span[1]]

                            for index in range(start_token_index, end_token_index + 1):
                                token_index_to_cluster_merged[index] = len(merged_clusters)

                    else:
                        print("Es wurden mehr als ein Cluster gefunden, was in Token übereinstimmt. Kein Merge.")
                        merged_clusters.append(cluster)
                        for span in combined_predictions[cluster_index]:
                            start_token_index = stored_info['subtoken_maps'][doc_key][span[0]]
                            end_token_index = stored_info['subtoken_maps'][doc_key][span[1]]

                            for index in range(start_token_index, end_token_index + 1):
                                token_index_to_cluster_merged[index] = len(merged_clusters)

            last_token_id = -1
            sentences = []
            for index, token_index in enumerate(stored_info['subtoken_maps'][doc_key]):
                if last_token_id == token_index: continue

                sentence_index = stored_info['sentence_map'][doc_key][index]

                if len(sentences) < sentence_index + 1: sentences.append([])
                word = {
                    'word': stored_info['tokens'][doc_key][token_index],
                    'clusters': [token_index_to_cluster[token_index]] if token_index in token_index_to_cluster else [],
                    'merged_clusters': [token_index_to_cluster_merged[token_index]] if token_index in token_index_to_cluster else [],
                }
                sentences[sentence_index].append(word)

                last_token_id = token_index

            output[doc_key] = {}
            output[doc_key]['sentences'] = sentences

        exclude_token_suffix = ".ex" if exclude_merge_tokens else ""
        output_path = join(config['data_dir'], f'hybrid.{language}.{max_seg_len}{exclude_token_suffix}.json')
        f = open(output_path, "w")
        f.write(json.dumps(output))
        f.close()


if __name__ == '__main__':
    config_name, saved_suffix, gpu_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out_file = sys.argv[4] if len(sys.argv) > 4 else None
    evaluate(config_name, gpu_id, saved_suffix, out_file)
