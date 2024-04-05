# Get Started

LLooM is currently designed as a Python package for computational notebooks. Follow the instructions on this page to get started with LLooM on your dataset. You can also refer to this [starter Colab Notebook]().

## Installation
First, install the LLooM Python package, available on PyPI as [`text_lloom`](https://pypi.org/project/text_lloom/). We recommend setting up a virtual environment with [venv](https://docs.python.org/3/library/venv.html#creating-virtual-environments) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).
```
pip install text_lloom
```

## LLooM Workbench
Now, you can use the LLooM package in a computational notebook! Create your notebook (i.e., with [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/)) and then follow the steps below.

### Import package
First, import the LLooM package:
```py
import text_lloom.workbench as wb
```

### OpenAI setup
LLooM uses the OpenAI API under the hood to support its core operators. You'll first need to locally set the `api_key` variable to use your own account.
```py
import os
os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY-HERE"
```

::: tip
LLooM provides (1) **cost estimation functions** that automatically run before operations that make calls to the OpenAI API and (2) **cost summary functions** to review tracked usage, but we encourage you to monitor usage on your account as always.
:::

### Create a LLooM instance
Then, after loading your data as a Pandas DataFrame, create a new LLooM instance. You will need to specify the name of the column that contains your input text documents (`text_col`). The ID column (`id_col`) is optional.
```py
l = wb.lloom(
    df=df,
    text_col="your_doc_text_col",
    id_col="your_doc_id_col",  # Optional
)
```

### Run concept generation
Next, you can go ahead and start the concept induction process by generating concepts. You can omit the `seed` parameter if you do not want to use a seed.
```py
await l.gen(seed="your_optional seed_term")
```

### Review and score concepts
Review the generated concepts and select concepts to inspect further:
```py
l.select()
```

In the output, each box contains the concept name, concept inclusion criteria, and representative example(s).
![LLooM select() function output](/media/ui/select_output.png)

Then, apply these concepts to the full dataset with `score()`. This function will score all documents with respect to each concept to indicate the extent to which the document matches the concept inclusion criteria.
```py
score_df = await l.score()
```

### Visualize concepts
Now, you can visualize the results in the main LLooM Workbench view. An interactive widget will appear when you run this function:
```py
l.vis()
```
Check out [Using the LLooM Workbench](./vis-guide.md) for a more detailed guide on the visualization components.
![LLooM vis() function output](/media/ui/vis_output.png)

#### Add slices (columns)
If you want to additionally slice your data according to a pre-existing metadata column in your dataframe, you can optionally provide a `slice_col`. Numeric or string columns are supported.
```py
l.vis(slice_col="n_likes")
```
By default, the concept matrix is normalized by **concept** (`norm_by="concept"`), meaning that the size of the circles in each concept row represents the fraction of examples *in that concept* that fall into each slice column. 
![LLooM vis() function output with slices](/media/ui/vis_output_slice.png)

You can also normalize by **slice** so that the size of circles in each slice column represents the fraction of examples *in that slice* that fall into each concept row.
```py
l.vis(slice_col="n_likes", norm_by="slice")
```

### Add manual concepts
You may also manually add your own custom concepts by providing a name and prompt. This will automatically score the data by that concept. Re-run the `vis()` function to see the new concept results.
```py
await l.add(
    name="your new concept name",
    prompt="your new concept prompt",
)
```

### Save LLooM instance
You can save your LLooM instance to a pickle file to reload at a later point.
```py
l.save(folder="your/path/here", file_name="your_file_name")
```

You can then reload the LLooM instance by opening the pickle file:
```py
import pickle
with open(f"your/path/here/your_file_name.pkl", "rb") as f:
    l = pickle.load(f)
```

### Export results
Export a summary of the results in Pandas Dataframe form. 
```py
export_df = l.export_df()
```
The dataframe will include the following columns:
- `concept`: The concept name
- `criteria`: The concept inclusion criteria
- `summary`: A generated summary of the examples that matched this concept
- `rep_examples`: A few representative examples for the concept from the concept generation phase
- `prevalence`: The proportion of documents in the dataset that matched this concept
- `n_matches`: The number of documents in the dataset that matched this concept
- `highlights`: An illustrative sample of n=3 highlighted quotes from documents that matched the concept that were relevant to the concept

## LLooM Operators
If you'd like to dive deeper and reconfigure the core operators used within LLooM (like the `Distill`, `Cluster`, and `Synthesize` operators), you can access the base functions from the `concept_induction` module:

```
import text_lloom.concept_induction as ci
```

Please refer to the [API Reference](../api/core) for details on each of the core LLooM operators.