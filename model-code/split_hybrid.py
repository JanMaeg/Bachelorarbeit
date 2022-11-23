import util
import json
from os.path import join
import numpy as np
import logging

logger = logging.getLogger(__name__)


def filter_cluster(clusters, start_index, end_index, normalize=False):
    final_clusters = []

    for cluster in clusters:
        filtered_cluster = [span for span in cluster if span[0] >= start_index and span[1] <= end_index]

        if normalize:
            filtered_cluster = np.array(filtered_cluster)
            filtered_cluster = filtered_cluster - start_index
            filtered_cluster = filtered_cluster.tolist()

        if len(filtered_cluster) > 0:
            final_clusters.append(filtered_cluster)

    return final_clusters


def split_document(samples, max_length=400):
    logger.info(f"Splitting documents into segment of length of around {max_length} subtokens.")
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
        current_sentence_index = 0  # The index of the sentence we currently analyze
        split_index = 0

        for sentence in sample['sentences']:
            token_count += len(sentence)
            split_sentences.append(sentence)
            total_sub_token_count += len(sentence)
            current_sentence_index += 1

            # At this point we have enough sub-tokens in our current set, so we split at this point.
            if token_count > max_length:
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

                split_sample[document_key] = {
                    "tokens": split_tokens,
                    "sentences": split_sentences,
                    "subtoken_map": (np.array(split_subtokens) - split_subtokens[0]).tolist(),
                    "subtoken_map_org": split_subtokens,
                    "speakers": sample["speakers"][start_sentence_index: current_sentence_index],
                    "sentence_map": (np.array(split_sentence_map) - split_sentence_map[0]).tolist(),
                    "clusters_org": filter_cluster(sample["clusters"], sub_token_index, end_index),
                    "clusters": filter_cluster(sample["clusters"], sub_token_index, end_index, True),
                    "start_index": sub_token_index,
                    "end_index": end_index,
                    "doc_key": document_key
                }

                # Proof that we match the correct speakers list o the sentences. I don't matter because
                # we don't have any speaker information for the datasets.
                assert len(split_sample[document_key]["speakers"][-1]) == len(
                    split_sample[document_key]["sentences"][-1])

                # Reset the variables for a new split set
                token_count = 0
                split_sentences = []
                sub_token_index = end_index + 1
                start_sentence_index = current_sentence_index
                split_index += 1

        assert total_split_sub_token_count == total_sub_token_count
        assert total_split_sub_token_count == len(sample['sentence_map'])
        assert total_token_count == len(sample["tokens"])

        splitted_documents.append(split_sample)
        logger.info(f"Splitted document {sample['doc_key']} into {len(split_sample)} segments")

    return splitted_documents


def get_clusters_for_subtoken_index(clusters, subtoken_index):
    cluster_ids = []

    for cluster_id, cluster in enumerate(clusters):
        for span in cluster:
            if span[0] <= subtoken_index <= span[1]:
                cluster_ids.append(cluster_id)
                continue

    return cluster_ids


def dump_to_file(documents, config, merged_clusters=None, file_name='gold_split.json', predictions = False):
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

    for doc_key in doc:
        last_index = -1

        for index, token_index in enumerate(doc[doc_key]["subtoken_map"]):
            if token_index == last_index:
                continue

            sentence_index = count_last_doc_sentences + doc[doc_key]["sentence_map"][index]
            if sentence_index >= len(sentences):
                sentences.append([])

            sub_token_index = index + doc[doc_key]['start_index'] if merged_clusters is not None else index
            clusters_key = "clusters" if not predictions else "predictions"

            clusters = get_clusters_for_subtoken_index(merged_clusters, sub_token_index) if merged_clusters is not None\
                else get_clusters_for_subtoken_index(doc[doc_key][clusters_key], index)

            word = {
                "sub_token_index": sub_token_index,
                "word": doc[doc_key]["tokens"][token_index],
                "clusters": clusters,
            }
            sentences[sentence_index].append(word)

            last_index = token_index

        count_last_doc_sentences = len(sentences)
        split_ends.append(count_last_doc_sentences)

    dump["sentences"] = sentences
    dump["split_ends"] = split_ends
    output_path = join(config['data_dir'], file_name)
    dump_file = open(output_path, "w")
    dump_file.write(json.dumps(dump))
    dump_file.close()


def main():
    config_name = "droc_test"

    config = util.initialize_config(config_name, create_dirs=False)

    language = config['language']
    max_seg_len = config['max_segment_len']
    path = join(config['data_dir'], f'test.{language}.{max_seg_len}.jsonlines')

    with open(path, 'r') as f:
        documents = [json.loads(line) for line in f.readlines()]
        splitted_documents = split_document(documents, 400)

        dump_to_file(splitted_documents, config)


if __name__ == "__main__":
    main()
