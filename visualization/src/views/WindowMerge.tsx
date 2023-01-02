import ClusterView from "../components/ClusterView/ClusterView";

import goldSplitData from "../data/string-matching/gold_split.json";
import predictedData from "../data/string-matching/predicted.german.128.json";
import mergedData from "../data/gold_split_overlapping.json";

const StringMatch = () => {
  return (
    <ClusterView
      goldData={goldSplitData}
      predictedData={predictedData}
      mergedData={mergedData}
      showClusterSelect
    />
  );
};

export default StringMatch;
