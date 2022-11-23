import Sentence from "./Sentence";
import { Text } from "@chakra-ui/react";

interface ClusterViewProps {
  goldData: {
    sentences: {
      sub_token_index: number;
      word: string;
      clusters: number[];
    }[][];
    split_ends: number[];
  };
  predictedData: {
    sentences: {
      sub_token_index: number;
      word: string;
      clusters: number[];
    }[][];
    split_ends: number[];
  };
  mergedData: {
    sentences: {
      sub_token_index: number;
      word: string;
      clusters: number[];
    }[][];
    split_ends: number[];
  };
}

const ClusterView = ({
  predictedData,
  goldData,
  mergedData,
}: ClusterViewProps) => {
  return (
    <div className="cluster-view">
      <div className="cluster-view__headline">
        <div />
        <div>
          <Text fontSize="xl">Gold</Text>
        </div>
        <div>
          <Text fontSize="xl">Merged Predictions</Text>
        </div>
        <div>
          <Text fontSize="xl">Split Predictions</Text>
        </div>
      </div>
      {goldData.sentences.map((sentence, index) => (
        <div className="cluster-view__row" key={index}>
          <div className="cluster-view__index">{index}</div>
          <Sentence sentence={sentence} />
          <Sentence
            start={mergedData.split_ends.includes(index)}
            sentence={mergedData.sentences[index]}
          />
          <Sentence
            start={predictedData.split_ends.includes(index)}
            sentence={predictedData.sentences[index]}
          />
        </div>
      ))}
    </div>
  );
};

export default ClusterView;
