# Get Started

LLooM is currently designed as a Python package for computational notebooks.

## Setup
### Installation
Install the LLooM Python package:
```
pip install text_lloom
```

### OpenAI setup
LLooM uses the OpenAI API under the hood to support its core operators. Set your OpenAI API key to use your account.
```
openai.api_key = "sk-YOUR-KEY-HERE"
```

## LLooM Workbench
### Import package
First, import the LLooM package:
```py
import text_lloom.workbench as wb
```

### Create a LLooM instance
Then, after loading your data as a Pandas DataFrame, create a new LLooM instance:
```py
l = wb.lloom(
    df=df,
    id_col="your_doc_id_col",
    text_col="your_doc_text_col",
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

Then, apply these concepts to the full dataset; this function will score all documents with respect to each concept to indicate the extent to which the document matches the concept inclusion criteria.
```py
score_df = await l.score()
```

### Visualize concepts
Now, you can visualize the results in the main LLooM Workbench view. An interactive widget will appear when you run this function:
```py
l.vis()
```

If you want to additionally slice your data according to a pre-existing metadata column, you can optionally provide a `slice_col`. Numeric or string columns are supported.
```py
l.vis(slice_col="n_likes")
```

By default, the concept matrix is normalized by **concept** (`norm_by="concept"`), meaning that the size of the circles in each concept row represents the fraction of examples *in that concept* that fall into each slice. You can also normalize by **slice** so that the size of circles in each slice column represents the fraction of examples *in that slice* that fall into each column.
```py
l.vis(slice_col="n_likes", norm_by="slice")
```

### Add manual concepts
You may also manually add your own custom concepts by providing a name and prompt. This will automatically score the data by that concept. Re-run the `vis()` function to see the new concept results
```py
await l.add(
    name="your new concept name",
    prompt="your new concept prompt",
)
```

### Save or export results
Save your LLooM instance to a pickle file to reload at a later point:
```py
l.save()
```

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
If you'd like to dive deeper and reconfigure the core operators used within LLooM (like the `Distill`, `Cluster`, and `Synthesize` operators), you can access the base functions from the concept_induction module:

```
import text_lloom.concept_induction as ci
```

Please refer to the [API Reference](../api/core) for details on each of the core LLooM operators.