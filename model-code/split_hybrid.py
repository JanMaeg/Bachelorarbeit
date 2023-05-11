import json
import logging
from os.path import join

import numpy as np

import util

logger = logging.getLogger(__name__)

STRING_MATCHING = "string_matching"
OVERLAPPING = "overlapping"
EMBEDDING = "embedding"
NEURAL = "neural"


def filter_cluster(clusters, start_index, end_index, normalize=False, correct_indices=False):
    final_clusters = []

    if correct_indices:
        final_clusters = {}

    for cluster_index, cluster in enumerate(clusters):
        filtered_cluster = [span for span in cluster if span[0] >= start_index and span[1] <= end_index]

        if normalize:
            filtered_cluster = np.array(filtered_cluster)
            filtered_cluster = filtered_cluster - start_index
            filtered_cluster = filtered_cluster.tolist()

        if len(filtered_cluster) > 0:
            if correct_indices:
                final_clusters[f"{cluster_index}"] = filtered_cluster
            else:
                final_clusters.append(filtered_cluster)

    return final_clusters


def split_document(samples, max_length=512, overlapping=True, overlapping_length=1):
    logger.info(f"Splitting documents into segment of length of less than {max_length} subtokens.")
    if overlapping:
        logger.info(f"Documents will overlap.")

    splitted_documents = []

    for sample in samples:
        token_count = 0
        sub_token_index = 0
        split_sentences = []
        split_sample = {}
        total_sub_token_count = 0
        total_token_count = 0
        total_split_sub_token_count = 0
        start_sentence_index = 0  # The index of the sentence where the current split starts
        split_index = 0

        sentence_index = 0
        overlapping_sub_token_count = 0

        while sentence_index < len(sample['sentences']):
            sentence = sample['sentences'][sentence_index]

            token_count += len(sentence)
            split_sentences.append(sentence)
            total_sub_token_count += len(sentence)

            # At this point we have enough sub-tokens in our current set, so we split at this point.
            if sentence_index == len(sample['sentences']) - 1 or token_count + len(sample['sentences'][sentence_index + 1]) > max_length:
                end_index = sub_token_index + token_count - 1
                split_subtokens = sample['subtoken_map'][sub_token_index: end_index + 1]

                if len(split_sample) != 0:
                    split_subtokens[0] += 1

                split_sentence_map = sample['sentence_map'][sub_token_index: end_index + 1]
                total_split_sub_token_count += len(split_subtokens)

                # The sub-list of sub-tokens we take should have the same length as the segments in total set.
                assert sum([len(sen) for sen in split_sentences]) == len(split_subtokens)

                # I am not sure why this happens but for all documents - except the first one - the index of 
                # first token has to be increased by 1. Otherwise, I would get the last token of the previous
                # split. 
                first_token_index = split_subtokens[0]
                last_token_index = split_subtokens[-1]
                split_tokens = sample["tokens"][first_token_index: last_token_index + 1]
                total_token_count += len(split_tokens)

                document_key = f"{sample['doc_key']}#{split_index}"

                logger.info(f"{document_key}: ${token_count} tokens")

                split_sample[document_key] = {
                    "tokens": split_tokens,
                    "sentences": split_sentences,
                    "subtoken_map": (np.array(split_subtokens) - split_subtokens[0]).tolist(),
                    "subtoken_map_org": split_subtokens,
                    "speakers": sample["speakers"][start_sentence_index: sentence_index + 1],
                    "sentence_map": (np.array(split_sentence_map) - split_sentence_map[0]).tolist(),
                    "sentence_map_org": split_sentence_map,
                    "clusters_org": filter_cluster(sample["clusters"], sub_token_index, end_index),
                    "clusters": filter_cluster(sample["clusters"], sub_token_index, end_index, True),
                    "clusters_correct_indices": filter_cluster(sample["clusters"], sub_token_index, end_index, True,
                                                               True),
                    "start_index": sub_token_index,
                    "end_index": end_index,
                    "doc_key": document_key,
                }

                assert len(split_sample[document_key]["speakers"]) == len(split_sentences)

                # Proof that we match the correct speakers list o the sentences. I don't matter because
                # we don't have any speaker information for the datasets.
                if len(split_sample[document_key]["speakers"]) > 1:
                    assert len(split_sample[document_key]["speakers"][-1]) == len(
                        split_sample[document_key]["sentences"][-1])

                # Reset the variables for a new split set
                token_count = 0
                split_sentences = []
                sub_token_index = end_index + 1
                start_sentence_index = sentence_index + 1
                split_index += 1

                if overlapping and sentence_index != len(sample['sentences']) - 1:
                    # We want to overlap with 2 sentences so instead of increasing by 1, we decrease by 1.
                    # This resets the counter in total by 2
                    # TODO: check if length of overlap is less than the length of the previous split
                    sentence_index -= overlapping_length - 1
                    start_sentence_index = sentence_index

                    # Reset the sub_token_index by the token count of the last 2 sentences because we want to add
                    # them again
                    overlapping_sub_token_count = len(sample['sentences'][sentence_index])

                    for i in range(1, overlapping_length):
                        overlapping_sub_token_count += len(sample['sentences'][sentence_index + i])

                    sub_token_index -= overlapping_sub_token_count
                else:
                    sentence_index += 1
            else:
                sentence_index += 1

        if not overlapping:
            assert total_split_sub_token_count == total_sub_token_count
            assert total_split_sub_token_count == len(sample['sentence_map'])
            assert total_token_count == len(sample["tokens"])

        splitted_documents.append(split_sample)
        logger.info(f"Splitted document {sample['doc_key']} into {len(split_sample)} segments")

    return splitted_documents


def get_clusters_for_subtoken_index(clusters, subtoken_index):
    cluster_ids = []

    if type(clusters) is dict:
        for cluster_id, cluster in clusters.items():
            for span in cluster:
                if span[0] <= subtoken_index <= span[1]:
                    cluster_ids.append(int(cluster_id))
                    continue
    else:
        for cluster_id, cluster in enumerate(clusters):
            for span in cluster:
                if span[0] <= subtoken_index <= span[1]:
                    cluster_ids.append(cluster_id)
                    continue

    return cluster_ids


def dump_to_file(documents, config, merged_clusters=None, file_name='gold.german.128.json', predictions=False,
                 method=STRING_MATCHING):
    if merged_clusters is not None:
        logger.info(f"Dump merged predictions of first document into file {file_name}")
    if predictions:
        logger.info(f"Dump predictions of first document into file {file_name}")
    else:
        logger.info(f"Dump raw split into file {file_name}")

    # We only dump the first document. More is not needed for a preview
    doc = documents[0]

    dump = {}
    sentences = []
    count_last_doc_sentences = 0

    split_ends = []
    split_starts = []

    skip_sentences_until = -1

    for doc_key in doc:
        split_starts.append(doc[doc_key]["sentence_map_org"][0])
        last_index = -1
        skipped_sentences = 0 if skip_sentences_until == - 1 else skip_sentences_until - \
                                                                  doc[doc_key]["sentence_map_org"][0] + 1
        last_skipped_index = 0

        for index, token_index in enumerate(doc[doc_key]["subtoken_map"]):
            if method is OVERLAPPING and \
                    skipped_sentences > -1 and \
                    doc[doc_key]["sentence_map"][index] <= skipped_sentences - 1:
                last_skipped_index = index
                continue

            if method is OVERLAPPING and skip_sentences_until > -1 and (
                    last_skipped_index + 1 == index or last_skipped_index + 2 == index):
                continue

            if token_index == last_index:
                continue

            sentence_index = count_last_doc_sentences + doc[doc_key]["sentence_map"][index]
            if method is OVERLAPPING and skipped_sentences > 0:
                sentence_index -= skipped_sentences
            if sentence_index >= len(sentences):
                sentences.append([])

            sub_token_index = index + doc[doc_key][
                'start_index'] if merged_clusters is not None or method is OVERLAPPING else index
            clusters_key = "clusters_correct_indices" if not predictions else "predictions"

            if method is NEURAL and merged_clusters is None:
                clusters_key = 'clusters'

            clusters = get_clusters_for_subtoken_index(merged_clusters, sub_token_index) if merged_clusters is not None \
                else get_clusters_for_subtoken_index(doc[doc_key][clusters_key], index)

            word = {
                "sub_token_index": sub_token_index,
                "word": doc[doc_key]["tokens"][token_index],
                "clusters": clusters,
            }
            sentences[sentence_index].append(word)

            last_index = token_index

        skip_sentences_until = doc[doc_key]["sentence_map_org"][-2]
        count_last_doc_sentences = len(sentences)
        split_ends.append(count_last_doc_sentences)

    dump["sentences"] = sentences
    dump["split_ends"] = split_ends
    dump["split_starts"] = split_starts
    if method is OVERLAPPING and merged_clusters is not None:
        dump["split_predictions"] = []
        for (_, document) in doc.items():
            index_corrected_clusters = []
            for clusters in document["predictions"]:
                index_corrected_clusters.append((np.array(clusters) + document["start_index"]).tolist())

            dump["split_predictions"].append(index_corrected_clusters)

    if method is OVERLAPPING:
        sub_folder = 'overlapping'
    elif method is STRING_MATCHING:
        sub_folder = 'string-matching'
    elif method is NEURAL:
        sub_folder = 'neural'
    else:
        sub_folder = 'embedding'

    output_path = join('/Users/jan/Documents/Studium/Bachelorarbeit/repository/visualization/src/data', sub_folder,
                       file_name)
    dump_file = open(output_path, "w")
    dump_file.write(json.dumps(dump))
    dump_file.close()


def main():
    config_name = "droc_test"
    overlapping = True

    config = util.initialize_config(config_name, create_dirs=False)

    language = config['language']
    max_seg_len = config['max_segment_len']
    path = join(config['data_dir'], f'test.{language}.{max_seg_len}.jsonlines')

    with open(path, 'r') as f:
        documents = [json.loads(line) for line in f.readlines()]
        splitted_documents = split_document(documents, 400, overlapping=overlapping)

        if overlapping:
            dump_to_file(splitted_documents, config, file_name='gold.german.128.json', method=OVERLAPPING)
        else:
            dump_to_file(splitted_documents, config, file_name='gold.german.128.json')


if __name__ == "__main__":
    main()
