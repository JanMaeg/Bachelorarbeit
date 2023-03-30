import SentenceWindow from "./SentenceWindow";
import csx from "classnames";
import {
  FormControl,
  FormLabel,
  Grid,
  GridItem,
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
  const [displayedSplit, setDisplayedSplit] = useState(0);
  const [displayedGoldCluster, setDisplayedGoldCluster] = useState(-1);
  const [displayedMergeCluster, setDisplayedMergeCluster] = useState(-1);

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
            padding: "10px",
          }}
        >
          <Grid templateColumns="repeat(3, 1fr)" gap={6}>
            <GridItem w="100%">
              <FormControl size="sm">
                <FormLabel>Highlighted Split</FormLabel>
                <NumberInput
                  size="sm"
                  defaultValue={0}
                  min={-1}
                  max={splits.length - 1}
                  value={displayedSplit}
                  onChange={(string, number) => {
                    setDisplayedSplit(number);
                  }}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
            </GridItem>{" "}
            <GridItem w="100%">
              <FormControl size="sm">
                <FormLabel>Gold cluster index</FormLabel>
                <NumberInput
                  size="sm"
                  defaultValue={-1}
                  min={-1}
                  max={splits.length - 1}
                  value={displayedGoldCluster}
                  onChange={(string, number) => {
                    setDisplayedGoldCluster(number);
                  }}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
            </GridItem>
            <GridItem w="100%">
              <FormControl size="sm">
                <FormLabel>Merged cluster index</FormLabel>
                <NumberInput
                  size="sm"
                  defaultValue={-1}
                  min={-1}
                  max={splits.length - 1}
                  value={displayedMergeCluster}
                  onChange={(string, number) => {
                    setDisplayedMergeCluster(number);
                  }}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
            </GridItem>
          </Grid>
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
            <Sentence
              onlyHighlightedCluster={displayedGoldCluster}
              sentence={sentence}
            />
            <Sentence
              onlyHighlightedCluster={displayedMergeCluster}
              start={mergedData.split_ends.includes(index)}
              sentence={mergedData.sentences[index]}
              overlapping={
                showClusterSelect &&
                displayedSplit > -1 &&
                splits[displayedSplit][0] <= index &&
                splits[displayedSplit][1] > index
              }
            />
            <SentenceWindow
              start={predictedData.split_ends.includes(index)}
              sentence={predictedData.sentences[index]}
              overlapping={
                showClusterSelect &&
                displayedSplit > -1 &&
                splits[displayedSplit][0] <= index &&
                splits[displayedSplit][1] > index
              }
              // clusters={predictedData.split_predictions[displayedSplit]}
            />
          </div>
        );
      })}
    </div>
  );
};

export default ClusterViewWindow;
