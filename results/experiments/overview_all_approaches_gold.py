import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['svg.fonttype'] = 'none'

string = [65.61, 88.93]
overlapping = [79.11, 92.22]
fast_text = [69.63, 88.92]
word_2_vec = [70.87, 91.20]
neural = [66.95, 90.36]


labels = ["String", "Overlapping", "fastText", "word2vec", "Neural"]

df = pd.DataFrame([string, overlapping, fast_text, word_2_vec, neural],
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
fig.savefig("f1_scores_overview_gold.svg", bbox_inches="tight")
plt.show()