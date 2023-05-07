import matplotlib.pyplot as plt
import numpy as np
import json

plt.rcParams['svg.fonttype'] = 'none'

thresholds = list(reversed(range(50, 96, 5)))

news_fastText = []
news_word2vec = []

droc_fastText = []
droc_word2vec = []

for threshold in thresholds:
    file_name_news_fastText = f"../logs/results_droc_final_news_embedding_500_fastText_{threshold}_.json"
    file_name_news_word2vec = f"../logs/results_droc_final_news_embedding_500_word2vec_{threshold}_.json"

    file_name_droc_fastText = f"../logs/results_droc_final_512_embedding_500_fastText_{threshold}_.json"
    file_name_droc_word2vec = f"../logs/results_droc_final_512_embedding_500_word2vec_{threshold}_.json"

    with open(file_name_droc_word2vec) as f:
        data = json.load(f)
        droc_word2vec.append(data["f1"] * 100)

    with open(file_name_droc_fastText) as f:
        data = json.load(f)
        droc_fastText.append(data["f1"] * 100)

    with open(file_name_news_word2vec) as f:
        data = json.load(f)
        news_word2vec.append(data["f1"] * 100)

    with open(file_name_news_fastText) as f:
        data = json.load(f)
        news_fastText.append(data["f1"] * 100)


plt.ylabel('CoNLL-F1-Score')
plt.xlabel('Similarity threshold')
plt.ylim(ymin=65, ymax=100)
plt.xticks(range(len(thresholds)), list(np.array(thresholds) / 100))
plt.plot(range(len(thresholds)), news_word2vec, label="TüBa-D/Z (word2vec)", marker="o")
plt.plot(range(len(thresholds)), news_fastText, label="TüBa-D/Z (fastText)", marker="o")
plt.plot(range(len(thresholds)), droc_fastText, label="DROC (fastText)", marker="o")
plt.plot(range(len(thresholds)), droc_word2vec, label="DROC (word2vec)", marker="o")
plt.legend()

plt.savefig('../graphics/embedding_similarity_threshold.svg', bbox_inches="tight")


plt.show()