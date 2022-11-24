import ClusterView from "../components/ClusterView/ClusterView";

import goldSplitData from "../data/gold_split.json";
import predictedData from "../data/predicted.german.128.json";
import mergedData from "../data/merged.german.128.json";

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
