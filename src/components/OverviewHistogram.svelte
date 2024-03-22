<script>
    // Component to render overview histogram
    // Imports
    import * as Plot from "@observablehq/plot";

    // Properties
    export let data;
    export let div;

    console.log("OverviewHistogram", div);

    function processData(data) {
        let processedData = [];
        data = JSON.parse(data);
        data.forEach((row) => {
            if (row["id"] == "All") {
                processedData.push({
                    name: row["concept"],
                    n: row["n"],
                });
            }
        });
        return processedData;
    }

    data = processData(data);
    let nConcepts = data.length;
    let histWidth = 100 * nConcepts;

    const plot = Plot.plot({
        marks: [
            Plot.gridY(),
            Plot.barY(data, {
                x: "name",
                y: "n",
                fill: "#A1D1FC",
                tip: true,
                sort: { x: "y", reverse: true, limit: 20 }
            }),
            Plot.axisX({
                label: "",
                lineWidth: 4,
            }),
            Plot.axisY({
                label: "Number of documents",
                // percent: true,
            }),
        ],
        marginBottom: 50,
        width: histWidth,
        height: 200,
    });
    div.replaceChildren(plot);
</script>

<div></div>

<style>
</style>
