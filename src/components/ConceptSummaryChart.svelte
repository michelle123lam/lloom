<script>
    // Component to render chart (viewing data by clusters) in notebook
    // Imports
    import * as Plot from "@observablehq/plot";

    // Properties
    export let rows = [];
    export let filterItems;
    export let sliceCol;
    export let domain;

    let container;

    function onChange(rows, filterItems) {
        let filteredRows = rows.filter((row) => {
            let keep = true;
            for (let [key, value] of Object.entries(filterItems)) {
                if (row[key] != value) {
                    keep = false;
                }
            }
            return keep;
        });
        rows = filteredRows;
        makePlot(rows);
    }

    function makePlot(data) {
        if (data.length > 0) {
            let curPlot = Plot.plot({
                x: {domain: domain},
                marks: [
                    Plot.gridY(),
                    Plot.barY(
                        data, 
                        Plot.groupX(
                            {y: 'count'},
                            {
                                x: 'id', 
                                fill: '#c1e0fd', 
                                tip: true, 
                            },
                        ),
                    ),
                    Plot.axisX({
                        label: sliceCol,
                        lineWidth: 8,
                        tickRotate: -45,
                        textAnchor: "end",
                    }),
                    Plot.axisY({
                        label: "Number of documents",
                    })
                ],
                marginBottom: 100,
                width: 300,
                height: 200,
            });

            // Replace the div with the plot
            container.replaceChildren(curPlot);
        }
    }

    $: onChange(rows, filterItems);
    
</script>

<div bind:this={container}/>

<style>
</style>
