import csx from "classnames";

interface SentenceProps {
  sentence: {
    sub_token_index: number;
    word: string;
    clusters: number[];
  }[];
  start?: boolean;
  overlapping?: boolean;
}

const PUNCTUATIONS = [",", ".", "!", "?", ":"];

const Sentence = ({
  sentence,
  start = false,
  overlapping = false,
}: SentenceProps) => {
  if (!sentence) return <div></div>;

  const elements = [];

  let previousClusterIndex = null;
  let currentCluster: string[] = [];
  for (let i = 0; i < sentence.length; i++) {
    const token = sentence[i];

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

export default Sentence;
