# Get Started

LLooM is currently designed as a Python package for computational notebooks. Follow the instructions on this page to get started with LLooM Workbench on your dataset. We suggest starting with this [**template Colab Notebook**](https://colab.research.google.com/github/michelle123lam/lloom/blob/main/docs/public/nb/24_04_LLooM_GettingStartedTemplate_v2.ipynb).

<a target="_blank" href="https://colab.research.google.com/github/michelle123lam/lloom/blob/main/docs/public/nb/24_04_LLooM_GettingStartedTemplate_v2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Installation
First, install the LLooM Python package, available on PyPI as [`text_lloom`](https://pypi.org/project/text_lloom/). We recommend setting up a virtual environment with [venv](https://docs.python.org/3/library/venv.html#creating-virtual-environments) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).
```
pip install text_lloom
```

## Setup
Now, you can use the LLooM package in a computational notebook! Create your notebook (i.e., with [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/)) and then follow the steps below.

### OpenAI setup
LLooM uses the OpenAI API under the hood to support its core operators (using GPT-3.5 and GPT-4). You'll first need to locally set the `OPENAI_API_KEY` variable to use your own account.
```py
import os
os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY-HERE"
```

::: tip
LLooM provides (1) **cost estimation functions** that automatically run before operations that make calls to the OpenAI API and (2) **cost summary functions** to review tracked usage, but we encourage you to monitor usage on your account as always.
:::

### Import package
Then, import the LLooM package:
```py
import text_lloom.workbench as wb
```

### Create a LLooM instance
After loading your data as a Pandas DataFrame, create a new LLooM instance. You will need to specify the name of the column that contains your input text documents (`text_col`). The ID column (`id_col`) is optional.
```py
l = wb.lloom(
    df=df,
    text_col="your_doc_text_col",
    id_col="your_doc_id_col",  # Optional
)
```

## Concept generation
Next, you can go ahead and start the concept induction process by generating concepts. The **seed term** can steer concept induction towards more specific areas of interest (e.g., social issues" for political discussion or "evaluation methods" for academic papers). You can omit the `seed` parameter if you do not want to use a seed.
```py
await l.gen(
    seed="your_optional_seed_term",  # Optional
)
```

We also provide a one-function **"auto" mode** that grants less control over the process, but simplifies the concept generation and scoring process into a single function. If you use the `gen_auto` function, you do not need to run concept scoring, but can directly proceed to visualize the results.
```py
score_df = await l.gen_auto(
    max_concepts=5,
    seed="your_optional_seed_term",  # Optional
)
```

## Concept scoring
### Review concepts
Review the generated concepts and select concepts to inspect further:
```py
l.select()
```

In the output, each box contains the concept name, concept inclusion criteria, and representative example(s).
![LLooM select() function output](/media/ui/select_output.png)

### Score concepts
Then, apply these concepts to the full dataset with `score()`. This function will score all documents with respect to each concept to indicate the extent to which the document matches the concept inclusion criteria.
```py
score_df = await l.score()
```

## Visualization
Now, you can visualize the results in the main LLooM Workbench view. An interactive widget will appear when you run this function:
```py
l.vis()
```
Check out [Using the LLooM Workbench](./vis-guide.md) for a more detailed guide on the visualization components.
![LLooM vis() function output](/media/ui/vis_output.png)

### Add slices (columns)
If you want to additionally slice your data according to a pre-existing metadata column in your dataframe, you can optionally provide a `slice_col`. Numeric or string columns are supported. Currently, numeric columns are automatically binned into quantiles, and string columns are treated as categorical variables.
```py
l.vis(slice_col="n_likes")
```
Optional parameters for slices:
- `max_slice_bins`: For numeric columns, the maximum number of bins to create (default=5)
- `slice_bounds`: For numeric columns, the manual bin boundaries to use (ex: [0, 10, 50, 100])

### Normalize counts
By default, the concept matrix shows the raw document counts. You can normalize by **concept**, meaning that the size of the circles in each concept row represents the fraction of examples *in that concept* that fall into each slice column. 
```py
l.vis(slice_col="n_likes", norm_by="concept")
```

You can also normalize by **slice** so that the size of circles in each slice column represents the fraction of examples *in that slice* that fall into each concept row.
```py
l.vis(slice_col="n_likes", norm_by="slice")
```
![LLooM vis() function output with slices](/media/ui/vis_output_slice.png)

## Manual concepts
You may also manually add your own **custom concepts** by providing a name and prompt. This will automatically score the data by that concept.
```py
await l.add(
    # Your new concept name
    name="Government Critique",
    # Your new concept prompt
    prompt="Does this text criticize government actions or policies?", 
)
```
Then, re-run the `vis()` function to see the new concept results.

## Saving and exporting

### Submit your results
::: tip 🖼️ ✨ Submit your work for a chance to be featured!
If you'd like to share what you've done with LLooM or would like your work featured in a gallery of results, please submit your LLooM instance with the `submit()` function! If your submission is selected, we'll reach out to you to follow up and hear more about your work with LLooM.
:::
To submit your results, you just need to run the following function:
```py
l.submit()
```
You will be prompted to provide a few more details: 
- **Email address**: Please provide an email address so that we can contact you if your work is selected.
- **Analysis goal**: Share as much detail as you'd like about your analysis: What data were you using? What questions were you trying to answer? What did you find?

### Save LLooM instance
You can save your LLooM instance to a pickle file to reload at a later point.
```py
l.save(folder="your/path/here", file_name="your_file_name")
```

You can then reload the LLooM instance by opening the pickle file:
```py
import pickle
with open("your/path/here/your_file_name.pkl", "rb") as f:
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

## Cost tracking
LLooM provides cost estimation functions as well as cost summary functions to review usage.

### estimate_gen_cost
`l.estimate_gen_cost(params=None, verbose=False)`

Estimates the cost of running `gen()` with the given parameters. If no parameters are provided, the function uses auto-suggested parameters. The function is automatically run within calls to `gen()` for the user to review before proceeding with concept generation.

### estimate_score_cost
`l.estimate_score_cost(n_concepts=None, verbose=False)`

Estimates the cost of running `score()` on the provided number of concepts. If `n_concepts` is not specified, the function uses the current number of active (selected) concepts. The function is automatically run within calls to `score()` for the user to review before proceeding with concept scoring.

### summary
`l.summary(verbose=True)`

Displays a **cumulative** summary of the (1) Total time, (2) Total cost, and (3) Tokens for the entire LLooM instance. 
- Total time: Displays the total time required for each operator. Each tuple contains the operator name and the timestamp at which the operation completed. 
- Total cost: Displays the calculated cost incurred by each operator (in US Dollars).
- Tokens: Displays the overall number of tokens used (total, in, and out)

Sample output:
```
Total time: 25.31 sec (0.42 min)
	('distill_filter', '2024-03-08-02-45-20'): 3.13 sec
	('distill_summarize', '2024-03-08-02-45-21'): 1.80 sec
	('cluster', '2024-03-08-02-45-25'): 4.00 sec
	('synthesize', '2024-03-08-02-45-42'): 16.38 sec


Total cost: $0.14
	('distill_filter', '2024-03-08-02-45-20'): $0.02
	('distill_summarize', '2024-03-08-02-45-21'): $0.02
	('synthesize', '2024-03-08-02-45-42'): $0.10


Tokens: total=67045, in=55565, out=11480
```

## Rate limits
Depending on the volume of data you are analyzing and the details of your OpenAI account/organization, you may run into OpenAI API [rate limits](https://platform.openai.com/docs/guides/rate-limits) (with respect to tokens per minute (TPM) or requests per minute (RPM)). LLooM provides several avenues to address rate limits.

### Modifying underlying models
By default, LLooM currently uses `gpt-3.5-turbo` for the Distill and Score operators, `gpt-4-turbo` for the Synthesize operator, and `text-embedding-3-large` for the Cluster operator. These values are set in [`workbench.py`](https://github.com/michelle123lam/lloom/blob/1f74f4569f8c8fc00f11852f1fcd136e35fc30df/text_lloom/src/text_lloom/workbench.py#L34-L37). However, users can specify different models for each of these operators within the LLooM instance.
```py {4-8}
l = wb.lloom(
    df=df,
    text_col="text",
    # Model specification
    distill_model_name = "gpt-3.5-turbo",
    embed_model_name = "text-embedding-3-large",
    synth_model_name = "gpt-4-turbo",
    score_model_name = "gpt-3.5-turbo",
)
```

### Customizing wait times
LLooM has built-in functionality for batching asynchronous requests to avoid rate limit issues. However, the necessary batch size and timing may vary across different users depending on details of their dataset and account. 

Thus, users can override the **number of requests in a batch** and the **length of time to wait** between batches with the `rate_limits` parameter. The current defaults are defined as `RATE_LIMITS` in [`llm.py`](https://github.com/michelle123lam/lloom/blob/24a7f5b1335311b90b4788608a3784a5d87e4482/text_lloom/src/text_lloom/llm.py#L34-L41). It may be helpful to refer to your organization's own [rate limits](https://platform.openai.com/account/limits) to set these values.

```py {4-10}
l = wb.lloom(
    df=df,
    text_col="text",
    # Rate limit parameter specification
    rate_limits={
        # Specify any custom parameters
        # Otherwise, they default to the RATE_LIMITS settings in llm.py
        # "model-name": (n_requests, wait_time_secs)
        "gpt-4-turbo": (40, 10),  
    }
)
```
- `n_requests`: number of requests allowed in one batch
- `wait_time_secs`: time period (in seconds) to wait before making more requests
- RPM (Requests per minute): `n_requests * (60 / wait_time_secs)`. In the above example, we specified 40 requests every 10 seconds, which means 240 requests per minute.

### Batching score operations
The `Score` operator in particular may run into rate limits because it initiates a higher volume of concurrent requests to score all documents for all selected concepts. By default, scoring is applied _individually_ for each concept and each document (`batch_size=1`) to produce more reliable scores. 

However, users can opt to increase the batch size, which will score multiple documents at a time for a given concept. This will reduce the number of unique API calls, but may sacrifice scoring reliability.

```py
score_df = await l.score(
    batch_size=5,
)
```


## LLooM Operators
If you'd like to dive deeper and reconfigure the core operators used within LLooM (like the `Distill`, `Cluster`, and `Synthesize` operators), you can access the base functions from the `concept_induction` module:

```
import text_lloom.concept_induction as ci
```

Please refer to the [API Reference](../api/core) for details on each of the core LLooM operators.