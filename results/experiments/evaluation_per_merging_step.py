import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['svg.fonttype'] = 'none'

with open('../evluation_per_merging_step_droc.json') as f:
    data = json.load(f)

with open('../evluation_per_mergin_step_news.json') as f:
    data_news = json.load(f)

f1s_droc = []
f1s_news = []

for results in data.values():
    f1 = results["f1"]
    f1s_droc.append(f1)

for results in data_news.values():
    f1 = results["f1"]

    if len(f1) < 19:
        f1s_news.append(f1)

max_length_droc = max([len(f1s) for f1s in f1s_droc])
max_length_news = max([len(f1s) for f1s in f1s_news])

# print(f1s_droc[:0])

meaned_f1_droc = []

for i in range(0, max_length_droc):
    f1_for_index_droc = []

    for val in f1s_droc:
        if len(val) >= i + 1:
            f1_for_index_droc.append(val[i] * 100)

    meaned_f1_droc.append(np.mean(f1_for_index_droc))

meaned_f1_news = []

for i in range(0, max_length_news):
    f1_for_index_news = []

    for val in f1s_news:
        if len(val) >= i + 1:
            f1_for_index_news.append(val[i] * 100)

    meaned_f1_news.append(np.mean(f1_for_index_news))

meaned_f1_news.insert(0, 100)
meaned_f1_droc.insert(0, 100)

plt.xticks(range(0, max_length_droc + 1))
plt.plot(range(0, max_length_droc + 1), meaned_f1_droc, label="DROC")
plt.plot(range(0, max_length_news + 1), meaned_f1_news, label="TüBa-D/Z")
plt.xlim(xmin=0, xmax=21)
plt.grid()
plt.legend()
plt.ylabel("Average CoNLL-F1-score")
plt.xlabel("Merging steps done")

plt.savefig('evaluation_over_time.svg', bbox_inches="tight")

plt.show()