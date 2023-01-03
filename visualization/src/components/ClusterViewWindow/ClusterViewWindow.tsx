import SentenceWindow from "./SentenceWindow";
import csx from "classnames";
import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Text,
} from "@chakra-ui/react";
import { useState } from "react";
import Sentence from "../ClusterView/Sentence";

interface ClusterViewWindowProps {
  goldData: {
    sentences: {
      sub_token_index: number;
      word: string;
      clusters: number[];
    }[][];
    split_ends: number[];
    split_starts?: number[];
  };
  predictedData: {
    sentences: {
      sub_token_index: number;
      word: string;
      clusters: number[];
    }[][];
    split_ends: number[];
    split_starts?: number[];
    split_predictions: number[][][][];
  };
  mergedData: {
    sentences: {
      sub_token_index: number;
      word: string;
      clusters: number[];
    }[][];
    split_ends: number[];
    split_starts?: number[];
  };
  showClusterSelect: boolean;
}

const ClusterViewWindow = ({
  predictedData,
  goldData,
  mergedData,
  showClusterSelect = false,
}: ClusterViewWindowProps) => {
  const [displayedCluster, setDisplayedCluster] = useState(0);

  const splits: number[][] = [];

  predictedData.split_starts?.forEach((split_start, index) =>
    splits.push([split_start, predictedData.split_ends[index]])
  );

  return (
    <div className="cluster-view">
      {showClusterSelect && (
        <div
          style={{
            position: "fixed",
            width: "100vw",
            background: "white",
            top: "0",
            display: "flex",
          }}
        >
          <NumberInput
            defaultValue={0}
            min={0}
            max={splits.length - 1}
            value={displayedCluster}
            onChange={(string, number) => {
              setDisplayedCluster(number);
            }}
          >
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </div>
      )}

      <div
        className={csx("cluster-view__headline", {
          "cluster-view__headline--down": showClusterSelect,
        })}
      >
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
      {goldData.sentences.map((sentence, index) => {
        return (
          <div className="cluster-view__row" key={index}>
            <div className="cluster-view__index">{index}</div>
            <Sentence sentence={sentence} />
            <Sentence
              start={mergedData.split_ends.includes(index)}
              sentence={mergedData.sentences[index]}
              overlapping={
                showClusterSelect &&
                splits[displayedCluster][0] <= index &&
                splits[displayedCluster][1] > index
              }
            />
            <SentenceWindow
              start={predictedData.split_ends.includes(index)}
              sentence={predictedData.sentences[index]}
              overlapping={
                showClusterSelect &&
                splits[displayedCluster][0] <= index &&
                splits[displayedCluster][1] > index
              }
              clusters={predictedData.split_predictions[displayedCluster]}
            />
          </div>
        );
      })}
    </div>
  );
};

export default ClusterViewWindow;
