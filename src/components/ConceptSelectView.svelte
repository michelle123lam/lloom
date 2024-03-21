<script>
    // Component to select concepts in notebook
    // Imports

    // Properties
    export let model;

    // Variables
    let concepts;

    let conceptsJson = model.get("data");
    if (conceptsJson) {
        concepts = JSON.parse(conceptsJson);
    }

    function handleCheck(c_id) {
        c_id = c_id.c_id;
        let concept_json = JSON.stringify(concepts);
        model.set("data", concept_json);
        model.save_changes();
    }
</script>

<div>
    <p class="header">Select concepts to score</p>
    <!-- Iterate over concepts dictionary -->
    {#if concepts}
        {#each Object.entries(concepts) as [c_id, c], i}
            <div class="concept-card">
                <div class="concept-detail">
                    <div class="left">
                        <input type="checkbox" id={c_id} name={c.name} bind:checked={c.active} on:change={() => handleCheck({c_id})}>
                        <label for={c_id}><b>{i + 1}: {c.name}</b></label>
                    </div>
                    <div class="mid">
                        <p>{c.prompt}</p>
                    </div>
                    <div class="right">
                        {#if c.examples}
                            <ul>
                                {#each c.examples as example}
                                    <li class="examples">"{example}"</li>
                                {/each}
                            </ul>
                        {/if}
                    </div>
                </div>
            </div>
        {/each}
    {/if}
</div>

<style>
    :global(.concept-card) {
        padding: 10px;
        margin: 5px 0; 
        border-radius: 10px;
        border: 1px solid #e6e6e6; 
        width: 95%;
    }

    :global(.concept-detail) {
        display: flex;
        flex-direction: row;
        justify-content: space-between; 
    }

    .left {
        width: 25%;
        display: inline-block;
    }
    .left input {
        float: left;
    }
    .left label {
        margin-left: 5px;
    }
    .mid {
        width: 40%
    }
    .right {
        float: right;
        width: 50%;
        padding-left: 10px;
    }

    .header {
        font-size: 12px;
        font-weight: bold;
        padding: 0px;
        margin: 0px;
    }
    .examples {
        font-size: 10px;
    }
    .concept-detail p, ul {
        padding: 0px 10px;
        margin: 0px;
    }
</style>
