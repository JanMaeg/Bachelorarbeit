import os

import matplotlib.pyplot as plt
import json
import numpy as np

plt.rcParams['svg.fonttype'] = 'none'

use_c2f = False

methods = [
    {"key": 'string_matching', "label": 'String', "values": [], "marker": "D"},
    {"key": 'embedding', "type": "word2vec", "label": 'word2vec', "values": [], "marker": "x"},
    {"key": 'embedding', "type": "fastText", "label": 'fastText', "values": [], "marker": "."},
    {"key": 'neural', "label": 'Neural', "values": [], "marker": "8"}
]
all_split_length = [500, 1000, 1500, 2000, 2500]

gold_suffix = "_c2f" if use_c2f else ""

for split_length in all_split_length:
    for method in methods:
        if method["key"] == "embedding":
            file_name = f'../logs/results_droc_final_512_{method["key"]}_{split_length}_{method["type"]}{gold_suffix}.json'
        elif method["key"] == "neural":
            file_name = f'../logs/results_droc_final_inc_512_{method["key"]}_{split_length}{gold_suffix}.json'
        else:
            file_name = f'../logs/results_droc_final_512_{method["key"]}_{split_length}{gold_suffix}.json'

        if not os.path.exists(file_name):
            print(file_name + " does not exist. Skipping...")
            continue

        with open(file_name) as f:
            data = json.load(f)
            method["values"].append(data["f1"])

fig, ax = plt.subplots(figsize=(5,3.5))

for method in methods:
    if len(method["values"]) > 0:
        ax.plot(all_split_length, np.array(method["values"]) * 100, label=method["label"], marker=method["marker"])

plt.ylim(ymin=50, ymax=85)
ax.set_xticks(all_split_length)
ax.set_xlabel('Maximum split length')
ax.set_ylabel('CoNLL-F1-Score')

plt.legend()

if use_c2f:
    plt.savefig("../graphics/evaluation_spit_length_c2f.svg", bbox_inches="tight")
else:
    plt.savefig("../graphics/evaluation_spit_length_gold.svg", bbox_inches="tight")

plt.show()