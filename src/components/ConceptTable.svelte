<script>
    // Component to render table in notebook
    // Imports
    import SvelteTable from "svelte-table";
    import { onDestroy, onMount } from "svelte";

    // Properties
    export let data;
    export let el;
    export let filterItems;

    let rows = [];
    let remount = true;
    let columns = [];

    onMount(() => {
        renderTable(data);
    });

    function renderTable(data_in) {
        let div = document.createElement("div");
        if (data_in != undefined) {
            let data_json = JSON.parse(data_in);
            rows = data_json;

            for (var i = 0; i < Object.keys(rows[0]).length; i++) {
                const key = Object.keys(rows[0])[i];
                if (key != "statement") {
                    columns.push({
                        key,
                        title: key,
                        value: (v) => v[key],
                        sortable: true,
                    });
                }
            }
            remount = false;
            setTimeout(() => (remount = true), 0);
        }
        el.appendChild(div);
    }
</script>

<div class="">
    {#if remount}
        <SvelteTable
            {columns}
            {rows}
            filterSelections={filterItems}
            expandSingle={true}
            rowKey="id"
            showExpandIcon={true}
        >
            <svelte:fragment slot="expanded" let:row
                >{row.statement}</svelte:fragment
            >
        </SvelteTable>
    {/if}
</div>

<style>
</style>
