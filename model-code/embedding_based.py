import numpy as np
import fasttext
import fasttext.util
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter
import logging

logger = logging.getLogger(__name__)


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


def merge_by_embedding(documents, use_gold_clusters=True, use_word2vec=False, threshold=50):
    if use_word2vec:
        model = KeyedVectors.load_word2vec_format("vectors_full.txt", binary=False, no_header=True)
    else:
        fasttext.util.download_model('de', if_exists='ignore')
        model = fasttext.load_model("cc.de.300.bin")

    cluster_str_key = "gold_cluster_str" if use_gold_clusters else "predictions_str"
    cluster_indices_key = "clusters" if use_gold_clusters else "predictions"

    merged_clusters = []

    threshold = threshold / 100

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
                    if highest_sim_value > threshold and np.argmax(
                            weights[:, highest_sim_index]) == cluster_index:
                        highlight_cell(highest_sim_index, cluster_index, color="red", linewidth=3)
                        logger.info(f"Cluster where merged by cosine similarity: {highest_sim_value}")
                        document_clusters[highest_sim_index] = np.concatenate(
                            (document_clusters[highest_sim_index], cluster))

                        document_clusters_indices[highest_sim_index] = np.concatenate(
                            (document_clusters_indices[highest_sim_index], cluster_indices_corrected)
                        )
                    else:
                        logger.info(
                            f"Cosine similarity doesn't pass threshold for any existing cluster. "
                            f"Adding new one {highest_sim_value}")
                        document_clusters.append(cluster)
                        document_clusters_indices.append(cluster_indices_corrected)

                # plt.show()

        merged_clusters.append({
            "str": document_clusters,
            "indices": document_clusters_indices
        })

    return merged_clusters
