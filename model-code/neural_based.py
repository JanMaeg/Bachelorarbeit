import itertools
from tensorize import Tensorizer, convert_to_torch_tensor

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

        enriched_tensor = tensorizer.tensorize_example(doc, False, False, split_starts, split_ends, predictions,
                                                       segments_per_split)
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
