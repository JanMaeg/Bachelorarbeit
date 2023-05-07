import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['svg.fonttype'] = 'none'

overview_f1_scores = [55.64, 51.11, 54.88, 55.94, 54.52, 64.72]

string = [55.64, 66.88]
overlapping = [51.11, 70.09]
fast_text = [54.88, 67.31]
word_2_vec = [55.94, 68.56]
neural = [54.52, 66.21]
baseline = [64.72, 78.79]

labels = ["String", "Overlapping", "fastText", "word2vec", "Neural", "Baseline"]

df = pd.DataFrame([string, overlapping, fast_text, word_2_vec, neural, baseline],
                 index=labels,
                 columns=pd.Index(['DROC', 'TüBa-D/Z']))

ax = df.plot(kind='bar',figsize=(10,4))

#ax = df.plot.bar()
for container in ax.containers:
    ax.bar_label(container)

ax.legend(loc='lower right')
ax.set_ylabel('CoNLL-F1-Score')
ax.set_xlabel('Method')
plt.ylim(ymin=0, ymax=100)
plt.yticks(range(0, 110, 10))

fig = ax.get_figure()
plt.xticks(rotation=45, ha='right')
fig.savefig("f1_scores_overview_c2f.svg", bbox_inches="tight")
plt.show()