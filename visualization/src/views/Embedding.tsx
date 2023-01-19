import ClusterView from "../components/ClusterView/ClusterView";

import goldSplitData from "../data/embedding/gold.german.128.json";
import predictedData from "../data/embedding/predicted.german.128.json";
import mergedData from "../data/embedding/merged.german.128.json";

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
