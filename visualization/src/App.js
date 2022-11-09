import { useState } from 'react'
import './App.css';

import data from "./data/test.german.512.json"
// import resultData from "./data/result.german.128.json"
// import splitData from "./split2.german.128.json"
import hybridData from "./data/hybrid.german.128.json"
import hybridExcludedData from "./data/hybrid.german.128.ex.json"


const PUNCTUATIONS = [",", ";", "!", ".", "?", ":"]

const CLUSTER_COLORS = [
    "#28C023",
    "#E74E5C",
    "#258385",
    "#18F840",
    "#A8B0AB",
    "#2A5276",
    "#2F3657",
    "#CF3E0F",
    "#B4A74F",
    "#7E79D8",
    "#A5E292",
    "#9661EB",
    "#D475C0",
    "#23909A",
    "#9C018A",
    "#7527C0",
    "#9B4848",
    "#4B2E4B",
    "#DE2E18",
    "#133766",
    "#703634",
    "#E538B4",
    "#3CC751",
    "#807A85",
    "#D3A3B8",
    "#FC4198",
    "#dccd6c",
    "#1fc2e3",
    "#e67491",
    "#d0b1c9",
    "#abc192",
    "#2b1678",
    "#9ab455",
    "#5a1105",
    "#846874",
    "#971033",
    "#31cd02",
    "#7c2089",
    "#9049dd",
    "#af8f41",
    "#f4a504",
    "#81b901",
    "#adf593",
    "#32d032",
    "#64dcb7",
    "#1e0162",
    "#574992",
    "#2f3540",
    "#ebebba",
    "#b866ec"
]

function getClusterOfToken(index, clusters) {
    const clustersOfToken = []

    clusters.forEach((clusterIndices, clusterIndex) => {
        clusterIndices.forEach(startEndIndices => {
            if (index >= startEndIndices[0] && index <= startEndIndices[1]) {
                clustersOfToken.push(clusterIndex)
            }
        })

    })

    return clustersOfToken
}

function getSentences(data) {
    const sentences = []
    const subtokens = data.sentences.flat()
    const subtokenMap = data.subtoken_map

    let currentToken = 0;

    for (let i = 0; i < subtokens.length; i++) {
        const sentenceIndex = data.sentence_map[i]

        if (i === subtokens.length - 1 || subtokenMap[i + 1] > subtokenMap[i]) {
            const token = data.tokens[currentToken]

            if (Array.isArray(sentences[sentenceIndex])) {
                sentences[sentenceIndex].push({word: token, clusters: getClusterOfToken(i, data.clusters)})
            } else {
                sentences[sentenceIndex] = [{word: token, clusters: getClusterOfToken(i, data.clusters)}]
            }

            currentToken++
        }
    }

    return sentences
}

function Sentence({words, showMergedClusters = false}) {

    const getStyle = (clusters) => {
        if (clusters.length === 0) return {}

        return {
            backgroundColor: CLUSTER_COLORS[clusters[0]]
        }
    }

    const getClusterId = (clusters) => {
        if (clusters.length === 0) return -1

        return clusters[0]
    }


    return (
        <p>{words.map((word, index) => {
            const clusters = showMergedClusters ? word.merged_clusters : word.clusters

            let clusterIdText = null
            const clusterId = getClusterId(clusters)
            if (clusterId !== -1) {
                clusterIdText = `(${clusterId})`
            }

            if (PUNCTUATIONS.includes(word.word) || index === words.length - 1) {
                return <span key={index} style={getStyle(clusters)}>{word.word} {clusterIdText}</span>
            }

            return [<span key={index}> </span>, <span key={`2-${index}`} style={getStyle(clusters)}>{word.word} {clusterIdText}</span>]
        })}</p>
    )

}

function Result({data, hybrid = false, showMergedClusters = false, showBoundaries = false }) {
    if (hybrid) {
        const documents = Object.values(data).map(data => data.sentences)

        return (
            <div className={`result ${showBoundaries && 'result--boundaries'}`}>
                {documents.map((document, index) => (
                    <div className="document" key={index}>
                        {document.map((sentence, sentenceIndex) => <Sentence words={sentence} key={sentenceIndex} showMergedClusters={showMergedClusters} />)}
                    </div>
                ))}
            </div>)
    }

    return (
        <div className="result document">
            {data.map((sentence, index) => <Sentence key={index} words={sentence}/>)}
        </div>
    )
}


function App() {
    const goldSentences = getSentences(data)

    const [showMergedClusters, setShowMergedClusters] = useState(false);
    const [excludeTokens, setExcludeTokes] = useState(false);
    const [showBoundaries, setShowBoundaries] = useState(true);

    const handleBoundariesChange = () => {
        setShowBoundaries(!showBoundaries);
    };
    
    const handleChange = () => {
        setShowMergedClusters(!showMergedClusters);
    };



    const handleExcludeTokensChange = () => {
        setExcludeTokes(!excludeTokens);
    };

    //const resultSentences = getSentences(resultData)
    //const splitSentences = getSentences(splitData)

    //console.log(`Amount mentions gold ${data.clusters.reduce((sum, cur) => sum + cur.length, 0)}`)
    //console.log(`Amount mentions without split ${resultData.clusters.reduce((sum, cur) => sum + cur.length, 0)}`)
    //console.log(`Amount mentions with split ${splitData.clusters.reduce((sum, cur) => sum + cur.length, 0)}`)

    return (
        <div>
            <div className="controls">
                <label>
                    <input
                        type="checkbox"
                        checked={showMergedClusters}
                        onChange={handleChange}
                    />
                    Show merged clusters
                </label> <br />
                <label>
                    <input
                        type="checkbox"
                        checked={showBoundaries}
                        onChange={handleBoundariesChange}
                    />
                    Show document boundaries
                </label> <br />
                <label>
                    <input
                        type="checkbox"
                        checked={excludeTokens}
                        onChange={handleExcludeTokensChange}
                    />
                    Don't use the following tokens for merge: ["er", "sie", "sich", "sein", "seinem", "ihre", "ihr", "Sie", "Ich", "ich", "ihm", "ihn", "seine", "ihres", "seinen", "ihrer", "ihrem", "seiner", "ihren"]
                </label>
            </div>
            <div className="App">
                {/*

                <Result data={resultSentences} />
                */}
                <Result data={goldSentences} />
                <Result data={excludeTokens ? hybridExcludedData : hybridData} hybrid showMergedClusters={showMergedClusters} showBoundaries={showBoundaries} />
            </div>
        </div>
    );
}

export default App;
