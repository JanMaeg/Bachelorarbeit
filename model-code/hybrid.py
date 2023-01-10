import os
import pickle

from split_hybrid import split_document, dump_to_file
from run import Runner
import sys
import util
from tensorize import Tensorizer, convert_to_torch_tensor
from os.path import join
import json
import itertools
import logging
import numpy as np
from metrics import CorefEvaluator

logger = logging.getLogger(__name__)

STRING_MATCHING = "string_matching"
OVERLAPPING = "overlapping"

METHOD = OVERLAPPING


def get_documents_with_predictions(documents, config, runner, model, out_file):
    tensorizer = Tensorizer(config)
    language = config['language']
    max_seg_len = config['max_segment_len']

    splitted_documents = split_document(documents, overlapping=(METHOD == OVERLAPPING))

    # To evaluate multiple documents with one call to the evaluate function we have change the structure of data
    # object. Our splitting function returns an array of dictionaries, where every split has an own key in the
    # dictionary. For the evaluation function we have to get rid of the dictionary/key-structure only have one big
    # array of dictionaries.
    flattened_documents = []
    for doc in splitted_documents:
        for doc_key in doc:
            flattened_documents.append(doc[doc_key])

    tensor_documents = itertools.chain(
        *(tensorizer.tensorize_example(doc, False, False) for doc in flattened_documents))
    tensor_documents = list(tensor_documents)

    torch_documents = [(doc_key, convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_documents]

    cache_path = join(config['data_dir'], f'predictions.{language}.{max_seg_len}')

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

            document[split_doc_key]["predictions_str"] = prediction_strings
        enriched_documents.append(document)

    return enriched_documents


def merge_by_overlapping(documents):
    merged_clusters = []

    for document in documents:
        document_clusters = []
        document_clusters_indices = []

        for index, split_doc_key in enumerate(document):
            logger.info(f"New split {index}")
            split = document[split_doc_key]

            # For the first split we simply take all clusters as they are
            # Obviously no need to merge them with anything.
            if index == 0:
                document_clusters = split["predictions_str"]
                document_clusters_indices = split["predictions"]
                continue

            for cluster_index, cluster in enumerate(split["predictions_str"]):
                logger.info("=========")
                logger.info(cluster)
                cluster_indices_corrected = np.array(split["predictions"][cluster_index]) + split["start_index"]
                logger.info(cluster_indices_corrected)

                max_overlaps = 0
                best_index = -1

                for existing_cluster_index, existing_cluster in enumerate(document_clusters_indices):
                    cluster_intersection = np.intersect1d(existing_cluster, cluster_indices_corrected)

                    if len(cluster_intersection) > max_overlaps:
                        max_overlaps = len(cluster_intersection)
                        best_index = existing_cluster_index

                if max_overlaps > 0:
                    logger.info(f"Cluster where merged with num overlaps: {max_overlaps / 2}")
                    logger.info(document_clusters[best_index])

                    document_clusters[best_index] = np.concatenate(
                        (document_clusters[best_index], cluster))

                    document_clusters_indices[best_index] = np.concatenate(
                        (document_clusters_indices[best_index], cluster_indices_corrected)
                    )
                else:
                    logger.info(f"No overlaps from the current cluster could be found in existing ones: {cluster}")
                    document_clusters.append(cluster)
                    document_clusters_indices.append(cluster_indices_corrected)

        merged_clusters.append({
            "str": document_clusters,
            "indices": document_clusters_indices
        })

    return merged_clusters


def merge_by_string_matching(documents):
    merged_clusters = []

    for document in documents:
        document_clusters = []
        document_clusters_indices = []

        for index, split_doc_key in enumerate(document):
            split = document[split_doc_key]

            # For the first split we simply take all clusters as they are
            # Obviously no need to merge them with anything.
            if index == 0:
                document_clusters = split["predictions_str"]
                document_clusters_indices = split["predictions"]
                continue

            for cluster_index, cluster in enumerate(split["predictions_str"]):
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
                    # if token in ["er", "sie", "sich", "sein", "seinem", "ihre", "ihr", "Sie", "Ich", "ich", "ihm",
                    # "ihn", "seine", "ihres", "seinen", "ihrer", "ihrem", "seiner", "ihren"]: continue

                    # The cluster that the token possibly matches the best with
                    best_cluster_match = -1

                    # The maximum number of times the current token appears in an already
                    # merged cluster
                    max_token_occurrences = -1

                    # Number of clusters that contain the current token
                    # We need this to check if more than one already merged cluster does contain the current token.
                    clusters_with_match_count = 0

                    for merged_cluster_index, merged_cluster in enumerate(document_clusters):
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
                        possible_merges_count += 1

                    if clusters_with_match_count == 1 and best_token_matches_count < max_token_occurrences:
                        best_token_matches_count = max_token_occurrences
                        possible_merge_cluster = best_cluster_match
                        possible_merges_count = 1
                        merge_token = token

                cluster_indices_corrected = np.array(split["predictions"][cluster_index]) + split["start_index"]

                if possible_merges_count == 1:
                    logger.debug(f"Cluster where merged by token: {merge_token}")
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

    enriched_documents = get_documents_with_predictions(documents, config, runner, model, out_file)
    enriched_documents = cluster_indices_to_tokens(enriched_documents)

    merged_clusters = []

    if METHOD == STRING_MATCHING:
        merged_clusters = merge_by_string_matching(enriched_documents)
    elif METHOD == OVERLAPPING:
        merged_clusters = merge_by_overlapping(enriched_documents)

    evaluator = CorefEvaluator()

    for doc_index, document in enumerate(enriched_documents):
        evaluator2 = CorefEvaluator()
        document_clusters = merged_clusters[doc_index]['indices']

        gold_clusters = documents[doc_index]['clusters']
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

        logger.info("=====================")
        metrics = {'Merge_Avg_Precision': p * 100, 'Merge_Avg_Recall': r * 100, 'Merge_Avg_F1': f * 100,
                   'Merge_Cur_Precision': p2 * 100, 'Merge_Cur_Recall': r2 * 100, 'Merge_Cur_F1': f2 * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))

    exclude_token_suffix = ".ex" if exclude_merge_tokens else ""
    dump_to_file(enriched_documents, config, None,
                 f'predicted.{language}.{max_seg_len}{exclude_token_suffix}.json', True,
                 overlapping=(METHOD == OVERLAPPING))

    dump_to_file(enriched_documents, config, merged_clusters[0]['indices'],
                 f'merged.{language}.{max_seg_len}{exclude_token_suffix}.json', False,
                 overlapping=(METHOD == OVERLAPPING))


def main():
    config_name, saved_suffix, gpu_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out_file = sys.argv[4] if len(sys.argv) > 4 else None
    evaluate(config_name, gpu_id, saved_suffix, out_file)


if __name__ == '__main__':
    main()
