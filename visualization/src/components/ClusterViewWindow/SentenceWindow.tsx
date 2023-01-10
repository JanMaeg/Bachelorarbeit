import csx from "classnames";

interface Sentence {
  sub_token_index: number;
  word: string;
  clusters: number[];
}

interface SentenceProps {
  sentence: Sentence[];
  start?: boolean;
  overlapping?: boolean;
  clusters: number[][][];
}

const PUNCTUATIONS = [",", ".", "!", "?", ":"];

const SentenceWindow = ({
  sentence,
  start = false,
  overlapping = false,
  clusters = [],
}: SentenceProps) => {
  if (!sentence) return <div></div>;

  const sentenceWithOverlappingClusters: Sentence[] = [];

  sentence.forEach((token) => {
    const updatedToken = { ...token };
    updatedToken.clusters = [];

    if (overlapping) {
      for (let i = 0; i < clusters.length; i++) {
        const cluster = clusters[i];

        cluster.forEach((mention) => {
          if (
            mention[0] <= token.sub_token_index &&
            token.sub_token_index <= mention[1]
          ) {
            updatedToken.clusters.push(i);
          }
        });
      }
    }

    sentenceWithOverlappingClusters.push(updatedToken);
  });

  const elements = [];

  let previousClusterIndex = null;
  let currentCluster: string[] = [];
  for (let i = 0; i < sentenceWithOverlappingClusters.length; i++) {
    const token = sentenceWithOverlappingClusters[i];

    const currentClusterIndex = token.clusters.length ? token.clusters[0] : -1;

    if (
      previousClusterIndex != null &&
      previousClusterIndex !== currentClusterIndex &&
      currentCluster.length > 0
    ) {
      elements.push(
        <span
          className={`cluster-view__cluster cluster-view__cluster--${previousClusterIndex}`}
          key={i}
        >
          {currentCluster.join(" ")}
          <span className="cluster-view__cluster-index">
            ({previousClusterIndex})
          </span>
        </span>
      );
      currentCluster = [];
    }

    if (currentClusterIndex !== -1) {
      currentCluster.push(token.word);

      if (i == sentence.length - 1) {
        elements.push(
          <span
            className={`cluster-view__cluster cluster-view__cluster--${previousClusterIndex}`}
            key={i}
          >
            {currentCluster.join(" ")}
            <span className="cluster-view__cluster-index">
              ({previousClusterIndex})
            </span>
          </span>
        );
      }
    } else {
      elements.push(token.word);
      if (
        i !== sentence.length - 1 &&
        !PUNCTUATIONS.includes(sentence[i + 1].word)
      )
        elements.push(" ");
    }

    previousClusterIndex = currentClusterIndex;
  }

  return (
    <div
      className={csx("cluster-view__sentence", {
        "cluster-view__sentence--start": start,
        "cluster-view__sentence--overlapping": overlapping,
      })}
    >
      <p>{elements}</p>
    </div>
  );
};

export default SentenceWindow;
