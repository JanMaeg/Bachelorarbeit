import argparse
import itertools
import json
import logging
import os
import pickle
from os.path import join

import util
from metrics import CorefEvaluator
from run import Runner
from split_hybrid import split_document
from tensorize import Tensorizer, convert_to_torch_tensor

from string_based import merge_by_string_matching
from overlapping_based import merge_by_overlapping
from embedding_based import merge_by_embedding
from neural_based import merge_by_neural_net

logger = logging.getLogger(__name__)

STRING_MATCHING = "string_matching"
OVERLAPPING = "overlapping"
EMBEDDING = "embedding"
NEURAL = "neural"


def get_documents_with_predictions(documents, config, runner, model, args):
    tensorizer = Tensorizer(config)
    language = config['language']
    max_seg_len = config['max_segment_len']

    splitted_documents = split_document(documents, max_length=args.split_length,
                                        overlapping=(args.method == OVERLAPPING),
                                        overlapping_length=args.overlapping_length)

    if args.use_c2f is None:
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

    cache_path = join(config['data_dir'],
                      f'predictions.{args.method}.{args.overlapping_length}.{language}.{max_seg_len}.{args.split_length}')

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
            out_file=None,
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


def update_evaluator(predicted_clusters, mention_to_cluster_id, gold_clusters, evaluator):
    mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
    gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
    mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)


def evaluate(args):
    config = util.initialize_config(args.config_name, create_dirs=False)
    runner = Runner(args.config_name, 0, skip_data_loading=True)
    model = runner.initialize_model(args.saved_suffix)
    model.eval_only = True
    exclude_merge_tokens = False

    language = config['language']
    max_seg_len = config['max_segment_len']
    path = join(config['data_dir'], f'test.{language}.{max_seg_len}.jsonlines')

    f = open(path, 'r')
    documents = [json.loads(line) for line in f.readlines()]
    documents = [document for document in documents if len(document['tokens']) > 350]

    enriched_documents = get_documents_with_predictions(documents, config, runner, model, args)
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

    logger.info(
        f"Removed {len(enriched_documents) - len(filtered_documents)} documents because only 1 split given. {len(filtered_documents)} documents left.")

    enriched_documents = filtered_documents

    logger.info([len(splits) for splits in enriched_documents])

    if args.method == STRING_MATCHING:
        merged_clusters = merge_by_string_matching(enriched_documents, use_gold_clusters=args.use_c2f is None,
                                                   exclude_pronouns=args.exclude_pronouns)
    elif args.method == OVERLAPPING:
        merged_clusters = merge_by_overlapping(enriched_documents, use_gold_clusters=args.use_c2f is None)
    elif args.method == EMBEDDING:
        merged_clusters = merge_by_embedding(enriched_documents, use_gold_clusters=args.use_c2f is None,
                                             use_word2vec=args.embedding_method == "word2vec",
                                             threshold=args.embedding_threshold)
    elif args.method == NEURAL:
        merged_clusters = merge_by_neural_net(enriched_documents, filtered_complete_documents, config, model, runner,
                                              None, use_gold_clusters=args.use_c2f is None)

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

    p, r, f = evaluator.get_prf()

    logger.info(f"Documents F1-Scores: {f1_total}")
    logger.info(f"Documents Recall: {r_total}")
    logger.info(f"Documents Precision: {p_total}")

    logger.info(f"{p} {r} {f}")

    results = {
        "f1": f,
        "precision": p,
        "recall": r,
        "documents_f1": f1_total,
        "documents_precision": p_total,
        "documents_recall": r_total,
        "method": args.method,
        "split_length": args.split_length,
        "config": args.config_name,
        "embedding_threshold": args.embedding_threshold
    }

    gold_suffix = "_c2f" if args.use_c2f is not None else ""
    overlap_suffix = f"_overlap_{args.overlapping_length}" if args.overlapping_length > 0 else ""

    if args.method == EMBEDDING:
        output_path = join(args.results_output,
                           f"results_{args.config_name}_{args.method}_{args.split_length}_{args.embedding_method}_{args.embedding_threshold}{gold_suffix}.json")
    else:
        output_path = join(args.results_output,
                           f"results_{args.config_name}_{args.method}_{args.split_length}{gold_suffix}{overlap_suffix}.json")
    dump_file = open(output_path, "w")
    dump_file.write(json.dumps(results))
    dump_file.close()

    # If you want to dump the gold and output entities into a file that can be loaded in the visualization,
    # uncomment the following lines.
    #
    # exclude_token_suffix = ".ex" if exclude_merge_tokens else ""
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True, default='droc_final_512',
                        help='Name of the experiment that should be run from the experiments.conf')
    parser.add_argument('--saved_suffix', type=str, required=True,
                        help='Suffix of the model that will be run')
    parser.add_argument('--method', type=str, required=True, default=STRING_MATCHING,
                        choices=[STRING_MATCHING, NEURAL, EMBEDDING, OVERLAPPING],
                        help='Merging method that should be applied')
    parser.add_argument('--split_length', type=int, default=512,
                        help='Maximum length of a split')
    parser.add_argument('--use_c2f', action=argparse.BooleanOptionalAction,
                        help='If splits should be predicted with coarse-to-fine model')
    parser.add_argument('--results_output', type=str, required=True,
                        help='Path were the results will be stored')
    parser.add_argument('--embedding_method', type=str, default="word2vec",
                        choices=["fastText", "word2vec"],
                        help='Type of word embeddings that are used')
    parser.add_argument('--embedding_threshold', type=int, default=95,
                        help='Required cosine-similarity to merged two entities')
    parser.add_argument('--overlapping_length', type=int, default=0,
                        help='Number of sentence that should overlap')
    parser.add_argument('--exclude_pronouns', action=argparse.BooleanOptionalAction,
                        help='If pronouns should be considered in string-based merging method')

    args = parser.parse_args()
    logger.info(args)

    evaluate(args)
