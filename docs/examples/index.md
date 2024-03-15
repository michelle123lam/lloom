---
template: demo
---


LLooM can assist with a range of data analysis goals—from preliminary exploratory analysis to theory-driven confirmatory analysis. Analysts can review LLooM concepts to interpret emergent trends in the data, but they can also author and edit concepts to actively seek out certain phenomena in the data. Concepts can be compared with existing metadata or other concepts to calculate statistics, generate plots, or train a model.

::: info EXAMPLE 1
#### Is social distrust related to partisan animosity on social media?
 1. Author a manual concept for `social distrust`.
 2. Review social media posts in the dataset that match the concept.
 3. Explore correlations between social distrust and partisan animosity.

TODO: replace with graphical representation of the above
<!-- ![The full LLooM Process](/media/lloom_process_full.svg) -->
:::

::: info EXAMPLE 2
#### What kinds of metaphors are used in toxic online comments?
1. Run concept induction on the dataset with a seed of `metaphors`.
2. Review the resulting concepts and compare metaphors with manual toxicity scores.
:::

::: info EXAMPLE 3
#### Has AI had a growing influence on HCI research?
1. Create a manual concept for `artificial intelligence (AI)`.
2. Review all paper abstracts that match the AI concept.
3. Plot the prevalence of AI mentions in papers over time.
:::

::: info EXAMPLE 4
#### Do authors use different rhetorical strategies when discussing benefits vs. risks in AI ethics statements?
1. Author concepts to separate excerpts related to benefits and those discussing risks.
2. Run concept induction on both datasets with a seed of `rhetorical strategies`.
3. Compare the concepts for the two datasets.
:::

::: info EXAMPLE 5
#### What critiques and notable points have students made in their commentaries for an assigned course reading?
1. Run concept induction on the dataset with a seed of `critiques and notable points`.
2. Review the resulting concepts and identify interesting ones worth discussing in class.
3. Export concepts and highlighted excerpts of commentaries to include in slides.
:::