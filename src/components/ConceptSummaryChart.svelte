<script>
    // Component to render chart (viewing data by clusters) in notebook
    // Imports
    import { onMount } from "svelte";
    import { Vega, VegaLite } from 'svelte-vega';

    // Properties
    export let rows = [];
    export let filterItems;
    export let sliceCol;
    let data;
    let viewVL;

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
        data = {
            rows: rows
        }
    }

    onChange(rows, filterItems);

    let specVL = {
		$schema: 'https://vega.github.io/schema/vega-lite/v5.json',
		description: 'A simple bar chart with embedded data.',
		data: {
			name: 'rows'
		},
		mark: {type: 'bar'},
        title: "Concept Prevalence",
        width: 200, 
        height: 150, 
		encoding: {
			y: { 
                field: 'concept_score_orig', 
                type: 'quantitative' , 
                aggregate: "sum",
                // scale: {
                //     domain: [0, 1]
                // },
                title: "Number of documents"
            },
			x: { 
                field: 'id', 
                type: 'ordinal',
                axis: {
                    labelLimit: 100,
                    labelAngle: -45
                },
                title: sliceCol,
                sort: null,
            },
            tooltip: {
                field: 'concept_score_orig',
                type: 'quantitative',
                aggregate: 'sum',
                // format: '.2f',
            }
		}
	};

    // $: viewVL ? console.log('Vega-Lite view: ', viewVL.data('rows')) : '';
    $: onChange(rows, filterItems);
    
</script>

<div class="chart">
    <VegaLite {data} spec={specVL} options={{actions: false}} bind:view={viewVL} />
</div>

<style>
</style>
