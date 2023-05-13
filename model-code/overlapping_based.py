import numpy as np
import logging

logger = logging.getLogger(__name__)


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
