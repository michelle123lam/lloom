<script setup>
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'
const { page, frontmatter } = useData()

// TEMP: pre=defined data for the demo
let testData = [
    "New year, new chance to sign up for a quality ACA healthcare plan. Before January 15, take some time to find a plan that works for you and #GetCovered in 2023 at",
    "It's been incredible to meet with so many New Yorkers today as we celebrate #MLKDay and as we work to honor the life and legacy of Reverend Dr. Martin Luther King, Jr. by continuing his march toward equality for all.",
    "Glad to be at the historic opening of the Alief Multi-Service Center; joined by Mayor @sylvesterturner and Councilmember @tiffanydeshellthomas . From swimming pools, to tennis courts, to skate parks, this facility will serve our city greatly for years to come.",
    "I obtained $1,800,000 for the Roe Road Extension Project in Paradise and $1,400,000 for the Cohasset Road Widening and Fire Safety Project to improve evacuation routes in those areas. These projects are focused on increasing road capacity to help people more quickly evacuate areas threatened by natural disasters, such as wildfires. This also aids first responders and emergency services to get to a disaster scene more expeditiously. These improvements to evacuation infrastructure will improve the safety and quality of life for the residents of Paradise and Butte County. Learning from previous disasters and expanding our ability to react and respond helps us prepare for potential new ones.",
    "The fatal beating of Tyre Nichols is horrifying. I'm devastated for his family and the Memphis community. We must fight for a world that ends this injustice and inhumane brutality at last.",
    "I am honored to continue serving on the Transportation and Infrastructure Committee Republicans. Solid infrastructure is critical to Florida's economy, which is dependent on moving goods and people efficiently and effectively. There's important work to be done in improving our infrastructure, and I look forward to working with Sam Graves, the members and applying my nearly 40 years of transportation policy and program experience to invest in our infrastructure system with creative funding ideas and innovative programs.",
    "I am grateful for the opportunity to represent the citizens of South Carolina's Second Congressional District and remain accessible to them.",
];

let concepts = [
    {
        "name": "Government Critique",
        "criteria": "Does this text criticize government actions or policies?",
        "summary": "Critique of government actions, policies, and officials, advocating for accountability, transparency, and reform.",
    },
    {
        "name": "Social and Economic Inequality",
        "criteria": "Does this text discuss social or economic disparities?",
        "summary": "Advocating for social justice, economic equality, healthcare access, and accountability in government and society.",
    },
    {
        "name": "Trust in Institutions",
        "criteria": "Does this text address trust or distrust in social or governmental institutions?",
        "summary": "Emphasizing trust in institutions through healthcare access, equality, disaster preparedness, combat readiness, and justice initiatives.",
    },
    {
        "name": "Policy and Healthcare Concerns",
        "criteria": "Does this text express concerns about healthcare policies or costs?",
        "summary": "Advocating for healthcare access, protecting abortion rights, lowering drug prices, and investigating federal agency corruption.",
    }
]
</script>

<template>
    <DefaultTheme.Layout>
        <template #doc-top>
            <div v-if="frontmatter.template === 'demo'">
                <div class="vp-doc"><h1>LLooM Examples</h1></div>
                <div class="full-width">
                    <!-- Jumbotron -->
                    <div class="jumbotron-wrapper">
                        <h2>Text Documents</h2>
                        <div class="jumbotron">
                            <div class="marquee marquee--hover-pause">
                                <div class="marquee__content">
                                    <div v-for="doc in testData" class="doc">
                                        {{ doc }}
                                    </div>
                                </div>

                                <div aria-hidden="true" class="marquee__content">
                                    <div v-for="doc in testData" class="doc">
                                        {{ doc }}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="arrow">
                        –
                        <img src="../../media/lloom.svg" alt="LLooM Logo">
                        →
                    </div>
                    <!-- Concepts -->
                    <div class="concepts-wrapper">
                        <h2>Concepts</h2>
                        <div class="concepts">
                            <div v-for="c in concepts" class="concept">
                                <h3>{{ c.name }}</h3>
                                <p><strong>Criteria:</strong> {{ c.criteria }}</p>
                                <p><strong>Summary:</strong> {{ c.summary }}</p>
                            </div>
                        </div>
                    </div>
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
    align-items: center;
}

.arrow {
    width: 10%;
    margin: auto;
    display: flex;
    flex-direction: row;
    align-items: center;
}

.arrow img {
    width: 80%;
    display: inline;
}

.concepts-wrapper {
    width: 45%;
    margin-right: 20px;
    float: right;   
}

.concepts {
    border-radius: 5px; 
    padding: 0 10px;
    opacity: 0.7;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.concept {
    margin: 10px auto;
    background-color: #c1e0fd;
    border: 1px solid #d5d5d5;
    border-radius: 5px;
    font-size: 11px;
    line-height: normal;
    text-align: left;
    padding: 10px;
    width: 100%;
}

.jumbotron-wrapper {
    width: 35%;
    height: 90%;
    margin-left: 20px;
}

.jumbotron {
    height: 100%;
    background-color: #f5f5f5;
    border: 1px solid gray;
    border-radius: 5px;
    padding: 0 10px;
    opacity: 0.7;
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