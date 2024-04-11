<script setup>
import { withBase } from 'vitepress'
import ConceptHist from './ConceptHist.vue';
import { ref, watch } from 'vue';

const props = defineProps({
    curConcepts: Object
})

const componentKey = ref(0);

const forceRerender = () => {
    componentKey.value += 1;
};

watch(() => props.curConcepts.data, async(newVal, oldVal) => {
    forceRerender();
})
</script>

<template>
    <div class="concepts-wrapper">
        <div class="concepts-hist">
            <ConceptHist :curConcepts="curConcepts" :key="componentKey" />
        </div>
        <div class="concepts">
            <div v-for="c in curConcepts.data" class="concept" :style="{backgroundColor: c.fill + '50'}">
                <h3>{{ c.name }}</h3>
                <p><strong>Criteria:</strong> {{ c.criteria }}</p>
                <p><strong>Summary:</strong> {{ c.summary }}</p>
            </div>
        </div>
    </div>
</template>

<style>
.concepts-wrapper {
    position: relative;
    width: 100%;
    float: right;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

.concepts {
    padding: 0 10px;
    margin: 5px auto;
}

.concepts-hist {
    margin: 10px auto;
}

.concept {
    margin: 5px auto;
    border: 1px solid #dedede;
    border-radius: 5px;
    font-size: 11px;
    line-height: normal;
    text-align: left;
    padding: 10px;
    width: 100%;
}
</style>