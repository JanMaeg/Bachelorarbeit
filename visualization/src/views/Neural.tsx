import ClusterView from "../components/ClusterView/ClusterView";

import goldSplitData from "../data/neural/gold.german.128.json";
import predictedData from "../data/neural/predicted.german.512.json";
import mergedData from "../data/neural/merged.german.512.json";

const Neural = () => {
  return (
    <ClusterView
      goldData={goldSplitData}
      predictedData={predictedData}
      mergedData={mergedData}
      showClusterSelect={false}
    />
  );
};

export default Neural;
