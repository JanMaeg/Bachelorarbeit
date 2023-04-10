import itertools
import json
import logging
import os
import pickle
import sys
from collections import Counter
from os.path import join

import fasttext
import fasttext.util
import matplotlib.pyplot as plt
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

import util
from metrics import CorefEvaluator
from run import Runner
from split_hybrid import split_document, dump_to_file
from tensorize import Tensorizer, convert_to_torch_tensor

logger = logging.getLogger(__name__)

STRING_MATCHING = "string_matching"
OVERLAPPING = "overlapping"
EMBEDDING = "embedding"
NEURAL = "neural"

METHOD = EMBEDDING
USE_GOLD_CLUSTER = False

EMBEDDING_THRESHOLD = 0.8


def get_documents_with_predictions(documents, config, runner, model, out_file, skip_predictions=True):
    tensorizer = Tensorizer(config)
    language = config['language']
    max_seg_len = config['max_segment_len']

    splitted_documents = split_document(documents, overlapping=(METHOD == OVERLAPPING))

    if skip_predictions:
        for index, docs in enumerate(splitted_documents):
            for doc_key, doc in docs.items():
                splitted_documents[index][doc_key]["predictions"] = []

        return splitted_documents

    # To evaluate multiple documents with one call to the evaluate function we have change the structure of data
    # object. Our splitting function returns an array of dictionaries, where every split has an own key in the
    # dictionary. For the evaluation function we have to get rid of the dictionary/key-structure only have one big
    # array of dictionaries.
    flattened_documents = []
    for doc in splitted_documents:
        for doc_key in doc:
            flattened_documents.append(doc[doc_key])

    tensor_documents = itertools.chain(
        *(tensorizer.tensorize_example(doc, False, True) for doc in flattened_documents))
    tensor_documents = list(tensor_documents)

    torch_documents = [(doc_key, convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_documents]

    cache_path = join(config['data_dir'], f'predictions.{METHOD.lower()}.{language}.{max_seg_len}')

    # For faster development I added here a caching for the predictions so that the model doesn't have
    # to prediction on every run (which takes some time).
    if os.path.exists(cache_path):
        # Load cached tensors if exists
        with open(cache_path, 'rb') as f:
            predictions = pickle.load(f)
            logger.info('Loaded predictions from cache')
    else:
        predictions = runner.evaluate(
            model,
            torch_documents,
            tensorizer.stored_info,
            0,
            official=False,
            conll_path=runner.config['conll_test_path'],
            out_file=out_file,
            hybrid=True
        )

        with open(cache_path, 'wb') as f:
            pickle.dump(predictions, f)

    # After generating predictions for all splits of all documents we append the predicted clusters to the
    # existing document data.
    last_doc_key = None
    document_index = 0
    for doc_key, predictions in predictions.items():
        doc_key_without_index = doc_key.split("#")[0]

        # For the first iteration
        if last_doc_key is None:
            last_doc_key = doc_key_without_index

        # When the document key without the index has changed we know that we moved to the next document.
        if doc_key_without_index != last_doc_key:
            document_index += 1

        splitted_documents[document_index][doc_key]["predictions"] = predictions
        last_doc_key = doc_key_without_index

    return splitted_documents


def cluster_indices_to_tokens(documents):
    enriched_documents = []

    for document in documents:
        for split_doc_key in document:
            prediction_strings = []

            for cluster in document[split_doc_key]["predictions"]:
                cluster_tokens = []

                for span in cluster:
                    start_token_index = document[split_doc_key]['subtoken_map'][span[0]]
                    end_token_index = document[split_doc_key]['subtoken_map'][span[1]]

                    tokens = document[split_doc_key]['tokens'][start_token_index: end_token_index + 1]
                    cluster_tokens.append(' '.join(tokens))

                prediction_strings.append(cluster_tokens)

            gold_prediction_strings = []

            for gold_cluster in document[split_doc_key]["clusters"]:
                gold_cluster_tokens = []

                for span in gold_cluster:
                    start_token_index = document[split_doc_key]['subtoken_map'][span[0]]
                    end_token_index = document[split_doc_key]['subtoken_map'][span[1]]

                    tokens = document[split_doc_key]['tokens'][start_token_index: end_token_index + 1]
                    gold_cluster_tokens.append(' '.join(tokens))

                gold_prediction_strings.append(gold_cluster_tokens)

            document[split_doc_key]["predictions_str"] = prediction_strings
            document[split_doc_key]["gold_cluster_str"] = gold_prediction_strings
        enriched_documents.append(document)

    return enriched_documents


def count_intersection(to_be_merged_cluster, existing_cluster):
    return np.intersect1d(to_be_merged_cluster, existing_cluster)


def merge_by_overlapping(documents, use_gold_clusters=False):
    predictions_str_key = "predictions_str" if not use_gold_clusters else "gold_cluster_str"
    predictions_key = "predictions" if not use_gold_clusters else "clusters"
    merged_clusters = []

    for index_2, document in enumerate(documents):
        document_clusters = []
        document_clusters_indices = []

        logger.info(f"Evaluating {index_2}")

        for index, split_doc_key in enumerate(document):
            split = document[split_doc_key]

            # For the first split we simply take all clusters as they are
            # Obviously no need to merge them with anything.
            if index == 0:
                document_clusters = split[predictions_str_key]
                document_clusters_indices = split[predictions_key]
                continue

            overlaps = []

            # We check for every existing cluster the overlap
            # 4 Existing Cluster A, B, C, D
            # 3 New cluster 1, 2, 3,
            #
            # Matrix 4 x 3
            #
            # A [ 0, 5, 6 ]
            # B [ 0, 5, 6 ]
            # C [ 0, 5, 6 ]
            # D [ 0, 5, 6 ]

            for existing_cluster_index, existing_cluster in enumerate(document_clusters_indices):
                local_overlaps = []

                for cluster_index, cluster in enumerate(split[predictions_str_key]):
                    cluster_indices_corrected = np.array(split[predictions_key][cluster_index]) + split["start_index"]
                    cluster_intersection = count_intersection(cluster_indices_corrected, existing_cluster)
                    local_overlaps.append(len(cluster_intersection))

                overlaps.append(local_overlaps)

            already_merged = []

            # After all overlaps have been calculated we find for every "new"/"to-be-merged" cluster the
            # best fitting "existing" cluster
            for cluster_index, cluster in enumerate(split[predictions_str_key]):
                best_index = None
                cluster_indices_corrected = np.array(split[predictions_key][cluster_index]) + split["start_index"]

                # Find the best index in that not has been merged until now
                while best_index in already_merged or best_index is None:
                    overlap_count_for_cluster = np.array(overlaps)[:, cluster_index]
                    overlap_count_for_cluster[already_merged] = 0
                    if max(overlap_count_for_cluster) == 0:
                        best_index = -1
                    else:
                        # Liefert index von bereits bestehendem Cluster (Im Beispiel wäre das hier ein Buchstabe)
                        best_index = np.argmax(overlap_count_for_cluster)

                # No overlaps in any entity that not has been merged already:
                if best_index == -1:
                    document_clusters.append(cluster)
                    document_clusters_indices.append(cluster_indices_corrected)
                    continue

                already_merged.append(best_index)
                document_clusters[best_index] = np.concatenate((document_clusters[best_index], cluster))
                document_clusters_indices[best_index] = np.concatenate(
                    (document_clusters_indices[best_index], cluster_indices_corrected))

        merged_clusters.append({
            "str": document_clusters,
            "indices": document_clusters_indices
        })

    return merged_clusters


def merge_by_string_matching(documents, use_gold_clusters=False):
    merged_clusters = []
    predictions_str_key = "predictions_str" if not use_gold_clusters else "gold_cluster_str"
    predictions_key = "predictions" if not use_gold_clusters else "clusters"

    for document in documents:
        document_clusters = []
        document_clusters_indices = []

        for index, split_doc_key in enumerate(document):
            split = document[split_doc_key]
            pre_merge_clusters = document_clusters

            # For the first split we simply take all clusters as they are
            # Obviously no need to merge them with anything.
            if index == 0:
                document_clusters = split[predictions_str_key]
                document_clusters_indices = split[predictions_key]
                continue

            # TODO: Try taken into account the number of occurences in the current entity
            for cluster_index, cluster in enumerate(split[predictions_str_key]):
                # Number of clusters that are possible for a merge.
                # E.g. if one cluster contains the string 5 times and another cluster contains another string also 5
                # times bother clusters are equally good for merging. possible_matches_count will be 2 in this case.
                # If this happens we won't merge the cluster because there is no way to decide which cluster works best.
                possible_merges_count = 0
                best_token_matches_count = 0
                possible_merge_cluster = -1
                merge_token = ''
                result = {}

                for token in cluster:
                    #if token in ["er", "sie", "sich", "sein", "seinem", "ihre", "ihr", "Sie", "Ich", "ich", "ihm",
#                                 "ihn", "seine", "ihres", "seinen", "ihrer", "ihrem", "seiner", "ihren"]: continue

                    # The cluster that the token possibly matches the best with
                    best_cluster_match = -1

                    # The maximum number of times the current token appears in an already
                    # merged cluster
                    max_token_occurrences = -1

                    # Number of clusters that contain the current token
                    # We need this to check if more than one already merged cluster does contain the current token.
                    clusters_with_match_count = 0

                    for merged_cluster_index, merged_cluster in enumerate(pre_merge_clusters):
                        # TODO: fix that merge in local cluster is not possible
                        # TODO: ignore the casing
                        token_count = np.count_nonzero(np.array(merged_cluster) == token)

                        # The current token does appear in an already merged cluster
                        # Because of this we increase the count here.
                        if token_count > 0:
                            clusters_with_match_count += 1

                        if token_count > 0 and max_token_occurrences < token_count:
                            max_token_occurrences = token_count
                            best_cluster_match = merged_cluster_index

                    result[token] = (best_cluster_match, max_token_occurrences, clusters_with_match_count)
                    if clusters_with_match_count == 1 \
                            and best_token_matches_count == max_token_occurrences \
                            and best_cluster_match != possible_merge_cluster:
                        # TODO: Irgendwie muss hier noch die Häufigkeit des tokens im to-be-merged entity berücksichtigt werden
                        possible_merges_count += 1

                    if clusters_with_match_count == 1 and best_token_matches_count < max_token_occurrences:
                        best_token_matches_count = max_token_occurrences
                        possible_merge_cluster = best_cluster_match
                        possible_merges_count = 1
                        merge_token = token

                cluster_indices_corrected = np.array(split[predictions_key][cluster_index]) + split["start_index"]

                if possible_merges_count == 1:
                    logger.info(f"Cluster where merged by token: {merge_token}")
                    document_clusters[possible_merge_cluster] = np.concatenate(
                        (document_clusters[possible_merge_cluster], cluster))

                    document_clusters_indices[possible_merge_cluster] = np.concatenate(
                        (document_clusters_indices[possible_merge_cluster], cluster_indices_corrected)
                    )
                elif possible_merges_count == 0:
                    logger.debug(f"No tokens from the current cluster could be found in any other cluster {cluster}")
                    document_clusters.append(cluster)
                    document_clusters_indices.append(cluster_indices_corrected)
                else:
                    logger.debug("Two or more merges with the same accuracy are possible. Skipping merge.")
                    document_clusters.append(cluster)
                    document_clusters_indices.append(cluster_indices_corrected)
                    logger.info("=====")
        merged_clusters.append({
            "str": document_clusters,
            "indices": document_clusters_indices
        })

    return merged_clusters


def map_word_frequency(document):
    return Counter(document)


def get_feature_vectors_word2vec(sentence1, sentence2, word_emb_model, use_sif=False):
    sentence1 = [f"{token.lower().encode()}" for token in sentence1.split() if
                 f"{token.lower().encode()}" in word_emb_model.key_to_index]
    sentence2 = [f"{token.lower().encode()}" for token in sentence2.split() if
                 f"{token.lower().encode()}" in word_emb_model.key_to_index]
    word_counts = map_word_frequency((sentence1 + sentence2))
    embedding_size = 300  # size of vector in word embeddings
    a = 0.001
    sentence_set = []
    for sentence in [sentence1, sentence2]:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        if sentence_length > 0:
            for word in sentence:
                if use_sif:
                    a_value = a / (a + word_counts[word])  # smooth inverse frequency, SIF
                    vs = np.add(vs, np.multiply(a_value, word_emb_model.get_vector(word)))
                else:
                    vs = np.add(vs, word_emb_model.get_vector(word))
            vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)
    return sentence_set


def get_feature_vectors_fasttext(sentence1, sentence2, word_emb_model, use_sif=False):
    sentence1 = sentence1.split()
    sentence2 = sentence2.split()

    word_counts = map_word_frequency((sentence1 + sentence2))
    embedding_size = 300  # size of vector in word embeddings
    a = 0.001
    sentence_set = []
    for sentence in [sentence1, sentence2]:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        if sentence_length > 0:
            for word in sentence:
                if use_sif:
                    a_value = a / (a + word_counts[word])  # smooth inverse frequency, SIF
                    vs = np.add(vs, np.multiply(a_value, word_emb_model.get_word_vector(word.lower())))
                else:
                    vs = np.add(vs, word_emb_model.get_word_vector(word))
            vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)
    return sentence_set


# https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def merge_by_embedding(documents, use_gold_clusters=True, use_word2vec=False):
    if use_word2vec:
        model = KeyedVectors.load_word2vec_format("vectors_full.txt", binary=False, no_header=True)
    else:
        fasttext.util.download_model('de', if_exists='ignore')
        model = fasttext.load_model("cc.de.300.bin")

    cluster_str_key = "gold_cluster_str" if use_gold_clusters else "predictions_str"
    cluster_indices_key = "clusters" if use_gold_clusters else "predictions"

    merged_clusters = []

    for document in documents:
        document_clusters = []
        document_clusters_indices = []

        for index, split_doc_key in enumerate(document):
            split = document[split_doc_key]
            current_predictions = split[cluster_str_key]

            # For the first split we simply take all clusters as they are
            # Obviously no need to merge them with anything.
            if index == 0:
                document_clusters = split[cluster_str_key]
                document_clusters_indices = split[cluster_indices_key]
                continue
            else:
                # figure(figsize=(15, 15), dpi=80)

                current_predictions_sentences = [" ".join(pre) for pre in current_predictions]
                previous_predictions_sentences = [" ".join(pre) for pre in document_clusters]

                weights = np.zeros([len(current_predictions), len(document_clusters)])

                for x, sentence1 in enumerate(current_predictions_sentences):
                    for y, sentence2 in enumerate(previous_predictions_sentences):
                        if use_word2vec:
                            feature_vectors = get_feature_vectors_word2vec(sentence1, sentence2, model)
                        else:
                            feature_vectors = get_feature_vectors_fasttext(sentence1, sentence2, model)

                        weights[x][y] = \
                            cosine_similarity(feature_vectors[0].reshape(1, -1), feature_vectors[1].reshape(1, -1))[0][
                                0]

                # plt.imshow(weights)
                # plt.yticks(np.arange(len(current_predictions)), [", ".join(pre) for pre in current_predictions])
                # plt.xticks(np.arange(len(document_clusters)), [", ".join(pre) for pre in document_clusters],
                #            rotation='vertical')
                # plt.subplots_adjust(bottom=0.6, left=0.6)
                # plt.colorbar()

                for cluster_index, _ in enumerate(weights):
                    cluster = split[cluster_str_key][cluster_index]
                    cluster_indices_corrected = np.array(split[cluster_indices_key][cluster_index]) + split[
                        "start_index"]

                    highest_sim_index = np.argmax(weights[cluster_index])
                    highest_sim_value = weights[cluster_index][highest_sim_index]

                    # If the highest cosine similarity is above the threshold both clusters will be merged
                    # Also check if the current index is also the highest possible for the target cluster
                    if highest_sim_value > EMBEDDING_THRESHOLD and np.argmax(
                            weights[:, highest_sim_index]) == cluster_index:
                        highlight_cell(highest_sim_index, cluster_index, color="red", linewidth=3)
                        logger.debug(f"Cluster where merged by cosine similarity: {highest_sim_value}")
                        document_clusters[highest_sim_index] = np.concatenate(
                            (document_clusters[highest_sim_index], cluster))

                        document_clusters_indices[highest_sim_index] = np.concatenate(
                            (document_clusters_indices[highest_sim_index], cluster_indices_corrected)
                        )
                    else:
                        logger.debug(
                            f"Cosine similarity doesn't pass threshold for any existing cluster. "
                            f"Adding new one {highest_sim_value}")
                        document_clusters.append(cluster)
                        document_clusters_indices.append(cluster_indices_corrected)

                # plt.show()
                # print("test")

        merged_clusters.append({
            "str": document_clusters,
            "indices": document_clusters_indices
        })

    return merged_clusters


def update_evaluator(predicted_clusters, mention_to_cluster_id, gold_clusters, evaluator):
    mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
    gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
    mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)


def merge_by_neural_net(enriched_documents, documents, config, model, runner, out_file, use_gold_clusters=False):
    tensorizer = Tensorizer(config)
    cluster_indices_key = "clusters" if use_gold_clusters else "predictions"

    tensors = []

    for doc, enriched_doc in zip(documents, enriched_documents):
        split_starts = [split['start_index'] for split in enriched_doc.values()]
        split_ends = [split['end_index'] for split in enriched_doc.values()]

        predictions = [split[cluster_indices_key] for split in enriched_doc.values()]

        segments_per_split = []

        for enriched_split in enriched_doc.values():
            segments_per_split.append(len(enriched_split['sentences']))

        enriched_tensor = tensorizer.tensorize_example(doc, False, False, split_starts, split_ends, predictions, segments_per_split)
        tensors.append(enriched_tensor)

    tensor_documents = itertools.chain(*tensors)
    tensor_documents = list(tensor_documents)
    torch_documents = [(doc_key, convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_documents]

    clusters = runner.evaluate(
        model,
        torch_documents,
        tensorizer.stored_info,
        0,
        official=True,
        conll_path=runner.config['conll_test_path'],
        out_file=out_file,
        hybrid=True
    )

    merged_clusters = []

    for doc_clusters in clusters.items():
        merged_clusters.append({
            "str": [],
            "indices": doc_clusters[1]
        })

    return merged_clusters


def evaluate(config_name, gpu_id, saved_suffix, out_file):
    config = util.initialize_config(config_name, create_dirs=False)
    runner = Runner(config_name, gpu_id, skip_data_loading=True)
    model = runner.initialize_model(saved_suffix)
    model.eval_only = True
    exclude_merge_tokens = False

    language = config['language']
    max_seg_len = config['max_segment_len']
    path = join(config['data_dir'], f'test.{language}.{max_seg_len}.jsonlines')

    f = open(path, 'r')
    documents = [json.loads(line) for line in f.readlines()]

    enriched_documents = get_documents_with_predictions(documents, config, runner, model, out_file, skip_predictions=USE_GOLD_CLUSTER)
    enriched_documents = cluster_indices_to_tokens(enriched_documents)

    merged_clusters = []

    filtered_documents = []
    filtered_complete_documents = []
    original_ids = []
    for document_id, splits in enumerate(enriched_documents):
        if len(splits) > 1:
            original_ids.append(document_id)
            filtered_documents.append(splits)
            filtered_complete_documents.append(documents[document_id])

    logger.info(f"Removed {len(enriched_documents) - len(filtered_documents)} documents because only 1 split given. {len(filtered_documents)} documents left.")

    enriched_documents = filtered_documents

    logger.info([len(splits) for splits in enriched_documents])


    if METHOD == STRING_MATCHING:
        merged_clusters = merge_by_string_matching(enriched_documents, use_gold_clusters=USE_GOLD_CLUSTER)
    elif METHOD == OVERLAPPING:
        merged_clusters = merge_by_overlapping(enriched_documents, use_gold_clusters=USE_GOLD_CLUSTER)
    elif METHOD == EMBEDDING:
        merged_clusters = merge_by_embedding(enriched_documents, use_gold_clusters=USE_GOLD_CLUSTER, use_word2vec=True)
    elif METHOD == NEURAL:
        merged_clusters = merge_by_neural_net(enriched_documents, filtered_complete_documents, config, model, runner, out_file, use_gold_clusters=USE_GOLD_CLUSTER)

    evaluator = CorefEvaluator()

    f1_total = []
    p_total = []
    r_total = []

    for doc_index, document in enumerate(enriched_documents):
        evaluator2 = CorefEvaluator()
        document_clusters = merged_clusters[doc_index]['indices']
        original_document_id = original_ids[doc_index]

        gold_clusters = documents[original_document_id]['clusters']
        predicted_clusters = []
        mention_to_cluster_id = {}

        for cluster_index, cluster in enumerate(document_clusters):
            predicted_clusters.append(tuple(tuple(m) for m in cluster))
            for mention in cluster:
                mention_to_cluster_id[tuple(mention)] = cluster_index

        update_evaluator(predicted_clusters, mention_to_cluster_id, gold_clusters, evaluator)
        update_evaluator(predicted_clusters, mention_to_cluster_id, gold_clusters, evaluator2)

        p, r, f = evaluator.get_prf()
        p2, r2, f2 = evaluator2.get_prf()

        f1_total.append(f2)
        p_total.append(p2)
        r_total.append(r2)

        logger.info("=====================")
        metrics = {'Merge_Avg_Precision': p * 100, 'Merge_Avg_Recall': r * 100, 'Merge_Avg_F1': f * 100,
                   'Merge_Cur_Precision': p2 * 100, 'Merge_Cur_Recall': r2 * 100, 'Merge_Cur_F1': f2 * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))

    logger.info(f"Documents F1-Scores: {f1_total}")
    logger.info(f"Documents Recall: {r_total}")
    logger.info(f"Documents Precision: {p_total}")

    exclude_token_suffix = ".ex" if exclude_merge_tokens else ""
    # dump_to_file(enriched_documents, config, None,
    #              f'predicted.{language}.{max_seg_len}{exclude_token_suffix}.json', True,
    #              method=METHOD)
    #
    # dump_to_file(enriched_documents, config, merged_clusters[0]['indices'],
    #              f'merged.{language}.{max_seg_len}{exclude_token_suffix}.json', False,
    #              method=METHOD)
    #
    # dump_to_file(enriched_documents, config, None,
    #              f'gold.{language}.{max_seg_len}{exclude_token_suffix}.json', False,
    #              method=METHOD)

def main():
    config_name, saved_suffix, gpu_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out_file = sys.argv[4] if len(sys.argv) > 4 else None
    evaluate(config_name, gpu_id, saved_suffix, out_file)


if __name__ == '__main__':
    main()
