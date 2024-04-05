<script>
    // Component to render chart (viewing data by concepts) in notebook
    // Imports
    import * as Plot from "@observablehq/plot";

    // Properties
    export let rows = [];
    export let filterItems;
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
            if (row["concept_score_orig"] < 1.0) {
                keep = false;
            }
            return keep;
        });
        rows = filteredRows;
        makePlot(rows);
    }

    function makePlot(data) {
        if (data.length > 0) {
            let curPlot = Plot.plot({
                y: {domain: domain},
                color: {
                    field: "concept",
                    type: "categorical",
                    scheme: "pastel2"
                },
                marks: [
                    Plot.gridX(),
                    Plot.barX(
                        data, 
                        Plot.groupY(
                            {x: 'count'},
                            {
                                y: 'concept', 
                                // fill: '#c1e0fd', 
                                fill: "concept",
                                tip: true, 
                            },
                        ),
                    ),
                    Plot.axisY({
                        label: "Concept",
                        lineWidth: 12,
                    }),
                    Plot.axisX({
                        label: "Number of documents",
                    })
                ],
                marginLeft: 150,
                width: 400,
                height: 250,
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
