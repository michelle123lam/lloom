<script>
import { withBase } from 'vitepress'
import * as Plot from "@observablehq/plot";

export default {
    props: {
        data: Array
    },
    mounted() {
        const plot = Plot.plot({
            marks: [
                Plot.barY(this.data, { x: 'name', y: 'frac', fill: '#679de5' }),
            ],
            x: {
                label: "Concepts",
                tickRotate: -30,
                tip: true,
            },
            y: {
                label: "Prevalence (%)",
                grid: true,
                percent: true,
            },
            marginBottom: 100,
            width: 400,
            height: 200,
        });
        this.$refs.view.replaceChildren(plot);
    }
}
</script>

<style>
.plot {
    margin-top: 1.5em;
    margin-bottom: 1em;
}
</style>

<template>
    <div class="plot" ref="view">
        <em>Loading Example...</em>
    </div>
</template>
