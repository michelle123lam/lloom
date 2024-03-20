<script setup>
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'
const { page, frontmatter } = useData()
import { reactive } from "vue";
import DemoConcepts from './DemoConcepts.vue';

import data from '../../public/data/demo_text.json'
import concepts from '../../public/data/demo_concepts.json'

const props = defineProps({
    curDataset: String,
    curScrollSpeed: String
})

// Variables
let curDataset = props.curDataset;
let curScrollSpeed = props.curScrollSpeed;
curScrollSpeed = (curScrollSpeed ? curScrollSpeed : "100s");
let curData;
const curSeed = reactive({ data: [] });
const curSeedOptions = reactive({ data: [] });
const curConcepts = reactive({ data: null });

// Initialize data
curData = data[curDataset];
curSeedOptions.data = Object.keys(concepts[curDataset]);
curSeed.data = curSeedOptions.data[0];
curConcepts.data = concepts[curDataset][curSeed.data];

// Set scroll speed
var r = document.querySelector(':root');
r.style.setProperty('--scroll-speed', curScrollSpeed);

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

</script>

<template>
    <div class="full-width demo">
        <!-- Jumbotron -->
        <div class="jumbotron-wrapper">
            <h2><b>Example Text Documents</b></h2>
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
        <!-- Results -->
        <div class="result-wrapper">
            <div class="logo-lloom">
                <img src="/media/lloom.svg" alt="LLooM Logo">
                <h2><b>Example Concepts</b></h2>
            </div>
            <!-- Seed selection -->
            <div id="seed-button-text">
                <p>Select seed</p>
            </div>
            <div class="seed-buttons">
                <button v-for="seedOpt in curSeedOptions.data" @click="updateSeed(seedOpt)" class="seed-button btn"
                    :class="{ active: isActiveSeed(seedOpt) }">
                    <span class="code">{{ seedOpt }}</span>
                </button>
            </div>
            <!-- Concepts -->
            <DemoConcepts :curConcepts="curConcepts" />
        </div>
    </div>
</template>

<style>

/* Text styles */
.demo h2, .demo h3, .demo p {
    margin: 0;
    border: none;
    padding: 0;
    line-height: normal;
}

.demo h2 {
    font-size: 16px;
}

.demo h3 {
    font-size: 14px;
    margin-bottom: 10px;
}

.demo p {
    line-height: normal;
    margin-bottom: 5px;
}

.code {
    font-family: monospace;
    font-size: 10px;
}

/* Layout styles */
.full-width {
    width: 100%;
    height: 75vh;
    text-align: center;
    border-radius: 5px;
    margin: 20px 0;
    display: flex;
    justify-content: space-between;
}

/* Result styles */
.result-wrapper {
    width: 65%;
    float: right;
}

.logo-lloom img {
    max-height: 20px;
}

.logo-lloom {
    font-size: 20px;
    margin-bottom: 20px;
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 5px;
    justify-content: center;
}

#seed-button-text {
    font-weight: bold;
    font-size: 12px;
    text-transform: uppercase;
    line-height: normal;
    text-align: center;
}

.seed-buttons {
    margin: 5px 0;
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    gap: 10px;
}

.seed-button {
    padding: 5px 8px;
    border-radius: 15px;
    background-color: white;
    border: 1px solid #d5d5d5;
    cursor: pointer;
    line-height: normal;
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


/* Jumbotron styles */
.jumbotron-wrapper {
    width: 30%;
    height: 100%;
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
    border-radius: 5px;
    padding: 10px 0;
    opacity: 0.7;
    position: absolute;
}

.doc {
    margin: 10px 0;
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
    animation: scroll var(--scroll-speed) linear infinite;
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