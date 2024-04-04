<script setup>
import { VPTeamMembers } from 'vitepress/theme'

const web_icon = {
     svg: '<svg class="MuiSvgIcon-root MuiSvgIcon-fontSizeMedium css-dhaba5" focusable="false" aria-hidden="true" viewBox="0 0 24 24" data-testid="PublicIcon"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2m-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39"></path></svg>'
};

const members = [
  {
    avatar: '/lloom/media/team/lam.jpg',
    name: 'Michelle S. Lam',
    title: 'PhD Candidate, Stanford',
    links: [
      { icon: web_icon, link: 'http://michelle123lam.github.io' },
      { icon: 'twitter', link: 'https://twitter.com/michelle123lam' },
      { icon: 'github', link: 'https://github.com/michelle123lam' },
    ]
  },
  {
    avatar: '/lloom/media/team/teoh.jpeg',
    name: 'Janice Teoh',
    title: 'Research Assistant, Stanford',
  },
  {
    avatar: '/lloom/media/team/landay.jpeg',
    name: 'James Landay',
    title: 'Professor, Stanford',
    links: [
      { icon: web_icon, link: 'https://www.landay.org/' },
      { icon: 'twitter', link: 'https://twitter.com/landay' },
    ]
  },
  {
    avatar: '/lloom/media/team/heer.jpeg',
    name: 'Jeffrey Heer',
    title: 'Professor, UW',
    links: [
      { icon: web_icon, link: 'https://homes.cs.washington.edu/~jheer/' },
      { icon: 'twitter', link: 'https://twitter.com/jeffrey_heer' },
    ]
  },
  {
    avatar: '/lloom/media/team/bernstein.jpeg',
    name: 'Michael S. Bernstein',
    title: 'Assoc Professor, Stanford',
    links: [
      { icon: web_icon, link: 'https://hci.stanford.edu/msb/' },
      { icon: 'twitter', link: 'https://twitter.com/msbernst' },
    ]
  },
];
</script>

# What is LLooM?

LLooM is a **data analysis tool** for **unstructured text** data, such as social media posts, paper abstracts, and articles. Manual text analysis is laborious and challenging to scale to large datasets, and automated approaches like topic modeling and clustering tend to focus on lower-level keywords that can be difficult for analysts to interpret.

By contrast, the LLooM algorithm turns unstructured text into meaningful **high-level concepts** that are defined by explicit inclusion criteria in **natural language**. For example, on a dataset of toxic online comments, while a BERTopic model outputs `"women, power, female"`, LLooM produces concepts such as `"Criticism of gender roles"` and `"Dismissal of women's concerns"`. We call this process **concept induction**: a computational process that produces high-level concepts from unstructured text.

The **LLooM Workbench** is an interactive text analysis tool that visualizes data in terms of the concepts that LLooM surfaces. With the LLooM Workbench, data analysts can inspect the automatically-generated concepts and author their own custom concepts to explore the data.

::: info LLooM Components
The LLooM Python package consists of two components:
- **`LLooM Workbench`**—a higher-level API for computational notebooks that surfaces interactive notebook widgets to inspect data by induced concepts.
- **`LLooM Operators`**—a lower-level API for the operators that underlie the LLooM algorithm.
:::

Check out the [Get Started](./get-started) page to try out LLooM.

## What can I do with LLooM?

Check out the [Examples](/examples/index) page to walk through case studies using LLooM.
- Briefly describe tasks you can achieve with LLooM
- Show screenshots/gif of the LLooM Workbench

![LLooM overview](/media/pull_figure.svg)

## How does LLooM work?
- Briefly describe the operators involved
- Add image summarizing the process

![The full LLooM Process](/media/lloom_process_full.svg)

## Learn more
LLooM is a research prototype! You can read much more about the project, the method, and a variety of evaluations in our CHI 2024 publication: [Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM]() by Michelle S. Lam, Janice Teoh, James Landay, Jeffrey Heer, and Michael S. Bernstein.

### Team members
<VPTeamMembers size="medium" :members="members" />

### Contact Us
Interested in the project or helping with future directions? We'd love to hear from you! Please feel free to contact Michelle at mlam4@cs.stanford.edu.

### Acknowledgements
Thank you to the Stanford HCI Group and UW Interactive Data Lab for feedback on early versions of this work. This work was supported in part by IBM as a founding member of the Stanford Institute for Human-centered Artificial Intelligence (HAI) and by NSF award IIS-1901386. Michelle was supported by a Stanford Interdisciplinary Graduate Fellowship.
