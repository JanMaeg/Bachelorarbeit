import matplotlib.pyplot as plt
import json

plt.rcParams['svg.fonttype'] = 'none'

max_overlap = 11
max_overlap_c2f = 11

f1s_gold = []
f1s_c2f = []

for overlap_length in range(1, max_overlap + 1):
    file_name = f'../logs/results_droc_final_overlapping_1500_overlap_{overlap_length}.json'

    with open(file_name) as f:
        data = json.load(f)
        f1s_gold.append(data["f1"])

for overlap_length in range(1, max_overlap_c2f + 1):
    file_name = f'../logs/results_droc_final_overlapping_1500_c2f_overlap_{overlap_length}.json'

    with open(file_name) as f:
        data = json.load(f)
        f1s_c2f.append(data["f1"] * 100)

plt.ylabel('CoNLL-F1-Score')
plt.xlabel('Number of overlapping sentences')
plt.ylim(ymin=45, ymax=60)
plt.xticks(range(1, max_overlap_c2f + 1))
#plt.plot(range(1, max_overlap + 1), f1s_gold, 'x', label='gold')
plt.plot(range(1, max_overlap_c2f + 1), f1s_c2f, 'x', label='c2f')
#plt.legend()
plt.savefig("../graphics/overlapping_size.svg", bbox_inches="tight")
plt.show()