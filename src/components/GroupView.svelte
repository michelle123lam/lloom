<script>
    // Component to render table in notebook
    // Imports
    import SvelteTable from "svelte-table";
    import GroupSummaryChart from "./GroupSummaryChart.svelte";
    import { onMount } from "svelte";

    // Properties
    export let data;
    export let dataLong;
    export let el;
    export let filterItems;
    export let sortBy;
    export let sortOrder;
    export let selectedTitle;
    export let selectedMetadata;

    let rows = [];
    let columns = [];
    let rowsLong = [];
    let remount = true;
    let domain = [];

    onMount(() => {
        renderTable(data, dataLong);
    });

    function renderTable(data_in, data_in_long) {
        if (data_in != undefined && data_in_long != undefined) {
            let div = document.createElement("div");
            let data_json = JSON.parse(data_in);
            rows = data_json;
            let data_long_json = JSON.parse(data_in_long);
            rowsLong = data_long_json;
            
            for (var i = 0; i < Object.keys(rows[0]).length; i++) {
                const key = Object.keys(rows[0])[i];
                    columns.push({
                        key,
                        title: key,
                        value: (v) => v[key],
                        sortable: true,
                        parseHTML: true,
                    });
                }
            remount = false;
            setTimeout(() => (remount = true), 0);
            el.appendChild(div);
            
            let fullDomain = rowsLong.map(d => d["concept"]);
            domain = [...new Set(fullDomain)];
        }
    }
</script>

<div>
    {#if selectedTitle}
        <h2>{selectedTitle}</h2>
    {/if}
    
    <div class="row">
        {#if selectedMetadata}
            <h3 class="card-title">SLICE DETAILS</h3>
            <div class="overview-card"> 
                <div class="overview-card-left-40">
                    {#each Object.entries(selectedMetadata) as [key, value]}
                        <p><b>{key}</b>: {value}</p>
                    {/each}
                </div>
                <div class="overview-card-right-60">
                    <GroupSummaryChart rows={rowsLong} {filterItems} {domain} />
                </div>
            </div>
        {/if}

        <h3 class="card-title">SLICE EXAMPLES</h3>
        <div class="highlight-card">
            {#if remount}
                <SvelteTable
                    {columns}
                    {rows}
                    sortBy={sortBy}
                    sortOrder={sortOrder}
                    filterSelections={filterItems}
                >
                </SvelteTable>
            {/if}
        </div>
    </div>
</div>

<style>
    .overview-card-left-40 {
        float: left !important;
        width: 40%;
        height: 100%;
    }
    .overview-card-right-60 { 
        float: right !important;
        width: 60%;
        height: 100%;
    }
</style>
