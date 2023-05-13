import numpy as np
import logging

logger = logging.getLogger(__name__)


def merge_by_string_matching(documents, use_gold_clusters=False, exclude_pronouns=False):
    merged_clusters = []
    predictions_str_key = "predictions_str" if not use_gold_clusters else "gold_cluster_str"
    predictions_key = "predictions" if not use_gold_clusters else "clusters"

    for document_index, document in enumerate(documents):
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
                    if exclude_pronouns and token.lower() in ["er", "sie", "sich", "sein", "seinem", "ihre", "ihr",
                                                              "Sie", "Ich", "ich", "ihm", "ihn", "seine", "ihres",
                                                              "seinen", "ihrer", "ihrem", "seiner", "ihren"]:
                        continue

                    # The cluster that the token possibly matches the best with
                    best_cluster_match = -1

                    # The maximum number of times the current token appears in an already
                    # merged cluster
                    max_token_occurrences = -1

                    # Number of clusters that contain the current token
                    # We need this to check if more than one already merged cluster does contain the current token.
                    clusters_with_match_count = 0

                    for merged_cluster_index, merged_cluster in enumerate(pre_merge_clusters):
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

                cluster_indices_corrected = np.array(split[predictions_key][cluster_index]) + split["start_index"]

                if possible_merges_count == 1:
                    # logger.info(f"Cluster where merged by token: {merge_token}")
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
