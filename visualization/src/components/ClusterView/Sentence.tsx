interface SentenceProps {
  sentence: {
    subtoken_map: number[];
    tokens: string[];
    split_index: number;
    start_sub_token_index: number;
  };
  clusters: number[][][];
}

const PUNCTUATIONS = [",", ".", "!", "?"];

const Sentence = ({ sentence, clusters }: SentenceProps) => {
  const elements = [];

  let previousTokenIndex = -1;
  let previousClusterIndex = null;
  let currentCluster: string[] = [];
  for (let i = 0; i < sentence.subtoken_map.length; i++) {
    const currentClusterIndex = clusters.findIndex((cluster) => {
      const filteredCluster = cluster.filter(
        (span) =>
          i + sentence.start_sub_token_index >= span[0] &&
          i + sentence.start_sub_token_index <= span[1]
      );
      return filteredCluster.length > 0;
    });

    const tokenIndex = sentence.subtoken_map[i];

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
      if (previousTokenIndex !== tokenIndex) {
        currentCluster.push(sentence.tokens[tokenIndex]);
        currentCluster.push(i + sentence.start_sub_token_index);
      }

      if (i == sentence.subtoken_map.length - 1) {
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
      if (previousTokenIndex !== tokenIndex) {
        elements.push(sentence.tokens[tokenIndex]);
        elements.push(i + sentence.start_sub_token_index);

        if (
          tokenIndex !== sentence.tokens.length - 1 &&
          !PUNCTUATIONS.includes(sentence.tokens[tokenIndex + 1])
        )
          elements.push(" ");
      }
    }

    previousClusterIndex = currentClusterIndex;
    previousTokenIndex = tokenIndex;
  }

  return (
    <div className="cluster-view__sentence">
      <p>{elements}</p>
    </div>
  );
};

export default Sentence;
