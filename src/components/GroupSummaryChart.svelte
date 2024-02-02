<script>
    // Component to render chart (viewing data by concepts) in notebook
    // Imports
    import { onMount } from "svelte";
    import { Vega, VegaLite } from 'svelte-vega';

    // Properties
    export let rows = [];
    export let filterItems;
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
        width: 150,
		encoding: {
			x: { 
                field: 'concept_score_orig', 
                type: 'quantitative' , 
                aggregate: "mean",
                scale: {
                    domain: [0, 1]
                },
                title: "Concept Prevalence"
            },
			y: { 
                field: 'concept', 
                type: 'nominal',
                axis: {
                    labelLimit: 100
                },
                sort: null,
                title: "Concept"
            },
            color: {
                field: 'concept',
                legend: null
            },
            tooltip:  {
                field: 'concept_score_orig',
                type: 'quantitative',
                aggregate: 'mean',
                format: '.2f',
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
