# What is LLooM?

LLooM is a **data analysis tool** for **unstructured text** data, such as social media posts, paper abstracts, and articles. Manual text analysis is laborious and challenging to scale to large datasets, and automated approaches like topic modeling and clustering tend to focus on lower-level keywords that can be difficult for analysts to interpret.

By contrast, the LLooM algorithm turns unstructured text into meaningful **high-level concepts** that are defined by explicit inclusion criteria in **natural language**. For example, on a dataset of toxic online comments, while a BERTopic model outputs `"women, power, female"`, LLooM produces concepts such as `"Criticism of gender roles"` and `"Dismissal of women's concerns"`. We call this process **concept induction**: a computational process that produces high-level concepts from unstructured text.

We introduce the **LLooM Workbench** as an interactive text analysis tool that visualizes data in terms of the concepts that LLooM surfaces. With the LLooM Workbench, data analysts can inspect the automatically-generated concepts and author their own custom concepts to explore the data.

::: info LLooM Components
The LLooM Python package consists of two components:
- **`LLooM Workbench`**—a higher-level API for computational notebooks that surfaces interactive notebook widgets to inspect data by induced concepts.
- **`LLooM Operators`**—a lower-level API for the operators that underlie the LLooM algorithm.
:::

Check out the [Get Started](./get-started) page to try out LLooM.

## What can I do with LLooM?
TODO
- Briefly describe tasks you can achieve with LLooM
- Show screenshots/gif of the LLooM Workbench

![LLooM overview](../media/pull_figure.svg)

## How does it work?
TODO
- Briefly describe the operators involved
- Add image summarizing the process

![The full LLooM Process](../media/lloom_process_full.svg)

## Learn more
LLooM is a research prototype! You can learn more about the project, the method, and a variety of evaluations in our CHI 2024 publication.

TODO
- Share some details about our team members
- Add acknowledgements
