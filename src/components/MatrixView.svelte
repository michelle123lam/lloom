<script>
    // Component to render matrix and table view in notebook
    // Imports
    import MatrixInner from "./MatrixInner.svelte";
    import ConceptView from "./ConceptView.svelte";
    import GroupView from "./GroupView.svelte";
    import OverviewHistogram from "./OverviewHistogram.svelte";

    // Properties
    export let model;
    export let el;

    // Variables: data
    let data;
    let dataItems;
    let dataItemsWide;
    let metadataOrig;
    let metadata;
    let numConcepts;
    let numSlices;

    // Variables: interaction settings
    let selectedMatrixElem;
    let filterItems;
    let sortBy;
    let sortOrder;
    let selectedTitle;
    let selectedMetadata;

    data = model.get("data");
    dataItems = model.get("data_items");
    dataItemsWide = model.get("data_items_wide");
    metadataOrig = model.get("metadata");
    metadata = JSON.parse(metadataOrig);
    numConcepts = Object.keys(metadata.concepts).length;
    numSlices = Object.keys(metadata.items).length;

    // // Set up enclosing div for overview and matrix
    let histDiv;
    let matrixDiv;

    function setToDefaultState() {
        // Reset
        selectedMatrixElem = null;
        filterItems = {};
        selectedTitle = null;
        selectedMetadata = null;
    }

    function handleMatrixEvent(event) {
        // Handles events from MatrixInner (sent via dispatch() function calls)
        let col = event.detail.col;
        let row = event.detail.row;
        let selection_type = event.detail.selection_type;
        if (selection_type == "cell") {
            // Set filtering by *selected cell*
            if (col == null) {
                setToDefaultState();
            } else {
                selectedMatrixElem = "cell";
                filterItems = { id: col, concept: row };
                selectedTitle = "Slice: " + col + ", Concept: " + row;
                let groupMetadata = metadata.items[col];
                let conceptMetadata = metadata.concepts[row];
                selectedMetadata = Object.assign(
                    groupMetadata,
                    conceptMetadata,
                );
            }
            // Reset sorting
            sortBy = null;
            sortOrder = 0; // none
        } else if (selection_type == "col") {
            let curSortOrder = event.detail.sortOrder;
            // Set filtering by *selected column*
            if (col == null) {
                setToDefaultState();
            } else {
                selectedMatrixElem = "group";
                filterItems = { id: col };
                selectedTitle = "Slice: " + col;
                selectedMetadata = metadata.items[col];

                sortBy = "concept_score";
                if (curSortOrder == 0) {
                    sortOrder = 0; // none
                } else if (curSortOrder == 1) {
                    sortOrder = 1; // ascending
                } else if (curSortOrder == 2) {
                    sortOrder = -1; // descending
                }
            }
        } else if (selection_type == "row") {
            let curSortOrder = event.detail.sortOrder;
            // Set filtering and sorting by *selected row*
            if (row == null) {
                setToDefaultState();
                sortOrder = 0; // none
            } else {
                selectedMatrixElem = "concept";
                filterItems = { concept: row };
                selectedTitle = "Concept: " + row;
                selectedMetadata = metadata.concepts[row];

                sortBy = "concept_score";
                if (curSortOrder == 0) {
                    sortOrder = 0; // none
                } else if (curSortOrder == 1) {
                    sortOrder = 1; // ascending
                } else if (curSortOrder == 2) {
                    sortOrder = -1; // descending
                }
            }
        }
    }
</script>

<div>
{#if numConcepts > 0}
    <h2 class="card-title">CONCEPT OVERVIEW</h2>
    <div id="histDiv" class="overview-hist" bind:this={histDiv}>
        {#key histDiv}
                {#if histDiv == undefined}
                    <p>Loading...</p>
                {:else}
                    <OverviewHistogram data={data} div={histDiv} />
                {/if}
        {/key}
    </div>

    <h2 class="card-title">CONCEPT MATRIX</h2>
    <div id="matrixWidget" class="matrix-widget">
        <div id="matrixDiv" class="matrix" bind:this={matrixDiv}>
            {#key matrixDiv}
                {#if matrixDiv == undefined}
                    <p>Loading...</p>
                {:else}
                    <MatrixInner
                        {data}
                        div={matrixDiv}
                        {numConcepts}
                        {numSlices}
                        on:message={handleMatrixEvent}
                    />
                    
                {/if}
            {/key}
        </div>
        {#if selectedMatrixElem == "cell" || selectedMatrixElem == "concept"}
        <div id="tableDiv" class="tables">
            <ConceptView
                data={dataItems}
                {el}
                {filterItems}
                {selectedTitle}
                {selectedMetadata}
            />
        </div>
        {:else if selectedMatrixElem == "group"}
        <div id="tableDiv" class="tables">
            <GroupView
                data={dataItemsWide}
                dataLong={dataItems}
                {el}
                {filterItems}
                {sortBy}
                {sortOrder}
                {selectedTitle}
                {selectedMetadata}
            />
        </div>
        {/if}
    </div>
{/if}
</div>

<style>
    :global(.overview-hist) {
        max-width: 100%;  
    }

    :global(.matrix-widget) {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
    }

    :global(.matrix-view) {
        min-height: 750px;
    }

    :global(.matrix) {
        float: left;
        /* max-width: 40%; */
        overflow-x: scroll;  
    }

    :global(.tables) {
        float: right;
        padding: 0 20px;
        overflow-x: scroll;
        max-width: 70%;
        border-radius: 10px;
        border: 1px solid #e6e6e6; 
        /* max-width: 60%; */
    }
    :global(.tables p) {
        font-size: 14px;
    }

    :global(h2) {
        font-size: 16px;
    }
    :global(h3) {
        font-size: 14px;
    }
</style>
