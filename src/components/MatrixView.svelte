<script>
    // Component to render matrix and table view in notebook
    // Imports
    import MatrixInner from "./MatrixInner.svelte";
    import ConceptView from "./ConceptView.svelte";
    import GroupView from "./GroupView.svelte";
    import { onMount } from "svelte";

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
    
    // Set up enclosing div for matrix
    let matrixDiv = document.createElement("div");
    matrixDiv.classList.add("matrix");
    el.appendChild(matrixDiv);
    el.classList.add("matrix-widget");


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
                filterItems = { id: col , concept: row };
                selectedTitle = "Slice: " + col + ", Concept: " + row;
                let groupMetadata = metadata.items[col];
                let conceptMetadata = metadata.concepts[row];
                selectedMetadata = Object.assign(groupMetadata, conceptMetadata)
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
                filterItems = { id: col};
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

{#if numConcepts > 0}
    {#key matrixDiv}
        <MatrixInner {data} div={matrixDiv} {numConcepts} {numSlices} on:message={handleMatrixEvent} />
    {/key}
    <div class="tables">
        {#if selectedMatrixElem == "cell" || selectedMatrixElem == "concept"}
            <ConceptView
                data={dataItems}
                {el}
                {filterItems}
                {selectedTitle}
                {selectedMetadata}
            />
        {:else if selectedMatrixElem == "group"}
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
        {/if}
    </div>
{/if}

<style>
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
    }

    :global(.tables) {
        float: right; 
        padding: 0 20px;  
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
