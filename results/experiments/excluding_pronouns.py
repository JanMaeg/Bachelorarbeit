import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['svg.fonttype'] = 'none'

droc_exclude = 73.16
droc_with = 66.34
news_exclude = 91.76
news_with = 88.85

news_with_c2f = 66.88
droc_with_c2f = 55.64
droc_exclude_c2f = 58.55
news_exclude_c2f = 68.97

experiments = ['DROC (gold)', 'TüBa-D/Z (gold)', 'DROC (c2f)', 'TüBa-D/Z (c2f)']

scores = {
    "include pronouns": [droc_with, news_with, droc_with_c2f, news_with_c2f],
    "exclude pronouns": [droc_exclude, news_exclude, droc_exclude_c2f, news_exclude_c2f]
}

x = np.arange(len(experiments))  # the label locations
width = 0.35  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in scores.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('CoNLL-F1-Score')
ax.set_xticks(x + width, experiments)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 110)

plt.savefig("../graphics/excluding_pronouns.svg", bbox_inches="tight")
plt.show()

