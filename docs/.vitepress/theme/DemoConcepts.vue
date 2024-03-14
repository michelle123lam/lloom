<script setup>
import { withBase } from 'vitepress'
import Example from './Example.vue';
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
    <!-- Concepts -->
    <div class="concepts-wrapper">
        <h2><b>Concepts</b></h2>
        <div class="concepts-hist">
            <Example :curConcepts="curConcepts" :key="componentKey" />
        </div>
        <div class="concepts">
            <div v-for="c in curConcepts.data" class="concept">
                <h3>{{ c.name }}</h3>
                <p><strong>Criteria:</strong> {{ c.criteria }}</p>
                <p><strong>Summary:</strong> {{ c.summary }}</p>
            </div>
        </div>
    </div>
</template>

<style>
.concepts-wrapper {
    width: 45%;
    height: 100%;
    margin-left: 10px;
    margin-right: 20px;
    float: right;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

.concepts {
    border-radius: 5px;
    padding: 0 10px;
}

.concepts-hist {
    max-height: 40%;
    overflow: scroll;
}

.concept {
    margin: 15px auto;
    background-color: #c1e0fd;
    border: 1px solid #dedede;
    border-radius: 5px;
    font-size: 11px;
    line-height: normal;
    text-align: left;
    padding: 10px;
    width: 100%;
}
</style>