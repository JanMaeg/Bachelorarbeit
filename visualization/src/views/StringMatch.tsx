import ClusterView from "../components/ClusterView/ClusterView";

import goldSplitData from "../data/string-matching/gold_split.json";
import predictedData from "../data/string-matching/predicted.german.128.json";
import mergedData from "../data/string-matching/merged.german.128.json";

const StringMatch = () => {
  return (
    <ClusterView
      goldData={goldSplitData}
      predictedData={predictedData}
      mergedData={mergedData}
      showClusterSelect={false}
    />
  );
};

export default StringMatch;
