import Sentence from "./Sentence";

interface ClusterViewProps {
  goldData: {
    sentences: {
      subtoken_map: number[];
      tokens: string[];
      split_index: number;
      start_sub_token_index: number;
    }[];
    clusters: number[][][];
  };
}

const ClusterView = ({ goldData }: ClusterViewProps) => {
  return (
    <div className="cluster-view">
      {goldData.sentences.map((sentence, index) => (
        <div className="cluster-view__row">
          <div className="cluster-view__index">{index}</div>
          <Sentence
            key={index}
            sentence={sentence}
            clusters={goldData.clusters}
          />
        </div>
      ))}
    </div>
  );
};

export default ClusterView;
