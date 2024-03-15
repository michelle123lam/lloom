<script setup>
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'
const { page, frontmatter } = useData()
import { reactive } from "vue";
import DemoConcepts from './DemoConcepts.vue';

import data from '../../public/data/demo_text.json'
import concepts from '../../public/data/demo_concepts.json'

// Variables
let curDataset;
let curData;
const curSeed = reactive({ data: [] });
const curSeedOptions = reactive({ data: [] });
const curConcepts = reactive({ data: null });

function updateData(dataset) {
    curDataset = dataset;
    curData = data[curDataset];
    curSeedOptions.data = Object.keys(concepts[curDataset]);

    // When changing datasets, use first seed
    curSeed.data = curSeedOptions.data[0];
    curConcepts.data = concepts[curDataset][curSeed.data];
}

function updateSeed(seed) {
    curSeed.data = seed;
    curConcepts.data = concepts[curDataset][curSeed.data];
}

function isActiveSeed(seed) {
    return seed === curSeed.data;
}

function isActiveDataset(dataset) {
    return dataset === curDataset;
}

const initialDataset = "Political social media";
updateData(initialDataset);

</script>

<template>
    <DefaultTheme.Layout>
        <template #doc-top>
            <div v-if="frontmatter.template === 'demo'">
                <div class="vp-doc">
                    <h1>LLooM Examples</h1>
                </div>
                <div class="dataset-buttons">
                    <div id="dataset-button-text">
                        <p>Select dataset</p>
                    </div>
                    <button v-for="dataset in Object.keys(data)" @click="updateData(dataset)" class="dataset-button btn"
                        :class="{ active: isActiveDataset(dataset) }">
                        {{ dataset }}
                    </button>
                </div>
                <div class="full-width">
                    <!-- Jumbotron -->
                    <div class="jumbotron-wrapper">
                        <h2><b>Text Documents</b></h2>
                        <div class="jumbotron-gradient"></div>
                        <div class="jumbotron">
                            <div class="marquee marquee--hover-pause">
                                <div class="marquee__content">
                                    <div v-for="doc in curData" class="doc">
                                        {{ doc }}
                                    </div>
                                </div>

                                <div aria-hidden="true" class="marquee__content">
                                    <div v-for="doc in curData" class="doc">
                                        {{ doc }}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <!-- Seed selection -->
                    <div class="arrow">
                        <div class="arrow-lloom">
                            –<img src="/media/lloom.svg" alt="LLooM Logo">→
                        </div>
                        <div class="seed-buttons">
                            <div id="seed-button-text">
                                <p>Select seed</p>
                            </div>
                            <button v-for="seedOpt in curSeedOptions.data" @click="updateSeed(seedOpt)"
                                class="seed-button btn" :class="{ active: isActiveSeed(seedOpt) }">
                                "{{ seedOpt }}"
                            </button>
                        </div>
                    </div>
                    <!-- Concepts -->
                    <DemoConcepts :curConcepts="curConcepts" />
                </div>
            </div>
        </template>
    </DefaultTheme.Layout>
</template>

<style>
.full-width {
    width: 100%;
    height: 75vh;
    text-align: center;
    border-radius: 5px;
    margin: 20px 0;
    display: flex;
    justify-content: space-between;
}

.arrow {
    width: 15%;
}

.arrow img {
    width: 80%;
    display: inline;
}

.arrow-lloom {
    font-size: 20px;
    display: flex;
    flex-direction: row;
    align-items: center;
    margin-bottom: 20px;
}

#seed-button-text {
    font-weight: bold;
    font-size: 12px;
    text-transform: uppercase;
    line-height: normal;
    text-align: center;
}

.seed-buttons {
    margin: 20px 0;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    align-items: center;
}

.seed-button {
    width: 100%;
    margin: 5px;
    padding: 5px 5px;
    border-radius: 5px;
    background-color: white;
    border: 1px solid #d5d5d5;
    font-size: 12px;
    cursor: pointer;
}

button.active {
    background-color: #679de5;
    color: white;
}

.dataset-buttons {
    margin: 20px 0;
    display: flex;
    justify-content: space-around;
    align-items: center;
}

#dataset-button-text {
    width: 75px;
    font-weight: bold;
    font-size: 12px;
    text-transform: uppercase;
    line-height: normal;
    text-align: right;
}

.dataset-button {
    margin: 5px;
    padding: 5px 10px;
    border-radius: 5px;
    background-color: white;
    border: 1px solid #d5d5d5;
    font-size: 12px;
    cursor: pointer;
}

.btn:hover {
    opacity: 0.75;
}

.jumbotron-wrapper {
    width: 30%;
    height: 100%;
    margin-left: 20px;
    position: relative;
}

.jumbotron-gradient {
    background: linear-gradient(white 2%, transparent 5%, transparent 95%, white 98%);
    z-index: 10;
    height: 95%;
    width: 100%;
    position: absolute;
}

.jumbotron {
    height: 95%;
    /* background-color: #f5f5f5; */
    /* border: 1px solid gray; */
    border-radius: 5px;
    padding: 0 10px;
    opacity: 0.7;
    position: absolute;
}

.doc {
    margin: 10px;
    background-color: white;
    border: 1px solid #d5d5d5;
    border-radius: 5px;
    font-size: 11px;
    line-height: normal;
    text-align: left;
    padding: 10px;
}

/* Marquee styles */
.marquee {
    --gap: 0.25rem;
    position: relative;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
    gap: var(--gap);
    height: 100%;
}

.marquee__content {
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    gap: var(--gap);
    min-height: 100%;
}

@keyframes scroll {
    from {
        transform: translateY(0);
    }

    to {
        transform: translateY(calc(-100% - var(--gap)));
    }
}

/* Pause animation when reduced-motion is set */
@media (prefers-reduced-motion: reduce) {
    .marquee__content {
        animation-play-state: paused !important;
    }
}

/* Enable animation */
.marquee__content {
    animation: scroll 100s linear infinite;
}

/* Pause on hover */
.marquee--hover-pause:hover .marquee__content {
    animation-play-state: paused;
}

/* Attempt to size parent based on content. Keep in mind that the parent width is equal to both content containers that stretch to fill the parent. */
.marquee--fit-content {
    max-width: fit-content;
}
</style>