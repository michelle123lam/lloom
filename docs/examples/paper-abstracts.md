# 3: Academic Paper Abstracts

<DemoLayout curDataset="Academic Paper Abstracts" curScrollSpeed="400s" />

## Task: Explore a research community's interests and impact
What impact has HCI research had on industry? A recent [large-scale measurement study](https://dl.acm.org/doi/10.1145/3544548.3581108) from Cao et al. investigated HCI's influence on industry through the lens of patent citations. This prior work used LDA topics to describe **trends among research that influenced patents**. We can use LLooM to **characterize research from the past 30 years** and further explore the **connections between HCI research topics, methods, and industry impact**.

## Dataset: ACM UIST paper abstracts, 1989-2018
We use the dataset from this prior research, which consists of paper abstracts from major HCI venues (CHI, CSCW, UIST, and UbiComp) from 1989 to 2018. We filter to UIST papers because they displayed an extremely outsized proportion of patent citations. We sought to better understand the nature of UIST research over time and potential factors underlying its high industry impact. To aid comparisons across time periods, we gathered a stratified random sample across each decade from 1989-1998, 1999-2008, and 2009-2018 with 70 papers from each decade for a total sample of 210 papers for this exploratory analysis.

## Analysis
:arrow_right: Try out analyzing this data with LLooM on this [Colab notebook]().