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
        }
    }
</script>

<div class="">
    {#if selectedTitle}
        <h2>{selectedTitle}</h2>
    {/if}
    
    <div class="row">
        <div class="left-col">
            {#if selectedMetadata}
                <h3 class="card-title">SLICE SUMMARY</h3>
                <div class="overview-card"> 
                    {#each Object.entries(selectedMetadata) as [key, value]}
                        <p><b>{key}</b>: {value}</p>
                    {/each}
                    <GroupSummaryChart rows={rowsLong} {filterItems} />
                </div>
            {/if}
        </div>

        <div class="right-col">
            <h3 class="card-title">SLICE EXAMPLES</h3>
            <div class="highlight-card-50">
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
</div>

<style>
    :global(.highlight-card-50) {
        padding-bottom: 20px;
        padding-inline: 20px;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        overflow-y: scroll;
        height: 90%;
    }
</style>
