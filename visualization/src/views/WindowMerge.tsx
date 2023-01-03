import ClusterViewWindow from "../components/ClusterViewWindow/ClusterViewWindow";

import goldSplitData from "../data/overlapping/gold.german.128.json";
import predictedData from "../data/overlapping/predicted.german.128.json";
import mergedData from "../data/overlapping/merged.german.128.json";

const StringMatch = () => {
  return (
    <ClusterViewWindow
      goldData={goldSplitData}
      predictedData={predictedData}
      mergedData={mergedData}
      showClusterSelect
    />
  );
};

export default StringMatch;
