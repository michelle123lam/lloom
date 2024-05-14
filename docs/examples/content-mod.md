# Content Moderation

<DemoLayout curDataset="Content Moderation" />

## Analysis
:arrow_right: Try out analyzing this data with LLooM on this [Colab notebook](https://colab.research.google.com/drive/1kVgs-rhj83egnCdpEEfYcYJKzCGUGda8?usp=sharing).

![LLooM, Content Moderation Notebook](/media/colab_tox.png)

## Task: Develop moderation policies for toxic content
Online content moderation has been a longstanding problem, and social media platforms devote significant resources to perform algorithmic content moderation. However, content moderation models are widely criticized for their errors and can often fail for marginalized communities. Given the substantial disagreement among the population on what constitutes toxic content, **how can we instead design models that account for the unique moderation needs of individual communities**? LLooM can help us **monitor emergent patterns of harm** in online communities, allowing us to identify and mitigate **gaps in content moderation models**.

## Dataset: Toxic social media posts
We use a dataset of social media posts (from Twitter, Reddit, and 4chan) that gathers a diverse set of annotators’ perspectives on content toxicity with ratings from 17,280 U.S. survey participants on over 100,000 examples. We further filtered to a set of posts related to _feminism_ as an example of a controversial topic area with a variety of user perspectives.
