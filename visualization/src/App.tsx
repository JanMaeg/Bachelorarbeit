import {useState} from "react";
import "./App.css";
import ClusterView from "./components/ClusterView/ClusterView";

import testData from "./data/hybrid_split.json";

function App() {
    console.log({testData})

    return (
        <div className="App">
            <ClusterView goldData={testData}/>
        </div>
    );
}

export default App;
