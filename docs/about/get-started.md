# Get Started

LLooM is currently designed as a Python package for computational notebooks.

## Installation
First, install the LLooM package:
```
pip install lloom_ai
```


## LLooM Workbench
### Import package
First, import the LLooM package:
```py
import lloom_ai.session as lloom
```

### Create a loom
Then, after loading your data as a Pandas DataFrame, create a new loom:
```py
l = lloom.Session(
    in_df=in_df,
    doc_id_col="your_doc_id_col",
    doc_col="your_doc_col",
)
```

### Run concept generation
Next, you can go ahead and start the concept induction process by generating concepts:
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

### Add manual concepts
You may also manually add your own custom concepts by providing a name and prompt. This will automatically score the data by that concept. Re-run the `vis()` function to see the new concept results
```py
await l.add(
    name="your new concept name",
    prompt="your new concept prompt",
)
```



## LLooM Operators
If you'd like to dive deeper and reconfigure the core operators used within LLooM (like the `Distill`, `Cluster`, and `Synthesize` operators), you can access the base functions from the concept_induction module:

```
import lloom_ai.concept_induction as ci
```

Please refer to the [API Reference](../api/core) for details on each of the core LLooM operators.