# LLooM Workbench

The LLooM Workbench is a higher-level API for computational notebooks that surfaces interactive notebook widgets to inspect data by induced concepts. It is defined as the `workbench` module within the `text_lloom` Python package and consists of the `lloom` class.

```py
import text_lloom.workbench as wb
```

# lloom
`lloom(df, text_col, id_col=None, distill_model_name="gpt-3.5-turbo", embed_model_name="text-embedding-3-large", synth_model_name="gpt-4-turbo", score_model_name="gpt-3.5-turbo", rate_limits={})`

A `lloom` instance manages a working session with a provided dataset. It allows the user to load their dataset and perform rounds of concept induction, concept scoring, and visualization.

**Parameters**:
- `df` _(pd.DataFrame)_: Text dataset that will be analyzed with LLooM. This dataframe must include a column that contains the input text documents. LLooM expects a dataframe where each row represents a distinct document (or unit of analysis). This is the primary text that will be analyzed with LLooM. The dataframe may also have other columns with additional metadata, which can be used for analysis with LLooM visualizations.
- `text_col` _(str)_: Name of the primary text column in the provided `df`.
- `id_col` _(str, optional, default: None)_: Name of a column with unique IDs for each row. If not provided, the system will assign an ID to each row.
- `distill_model_name` _(str, optional, default: "gpt-3.5-turbo")_: Name of the OpenAI model to use for the Distill operators (filter and summarize).
- `embed_model_name` _(str, optional, default: "text-embedding-3-large")_: Name of the OpenAI embedding model to use for the Cluster operator.
- `synth_model_name` _(str, optional, default: "gpt-4-turbo")_: Name of the OpenAI model to use for the Synthesize operator.
- `score_model_name` _(str, optional, default: "gpt-3.5-turbo")_: Name of the OpenAI model to use for the Score operator.
- `rate_limits` _(Dict, optional, default: {})_: An optional dictionary specifying a mapping from an OpenAI model to its associated rate-limit parameters, a tuple of the form (n_requests, wait_time_secs), where n_requests indicates the number of requests allowed in one batch and wait_time_secs indicates the length of time (in seconds) to wait between batches. Example: `{ "gpt-4-turbo": (40, 10) }`. If not specified, defaults to the values defined as `RATE_LIMITS` in [`llm.py`](https://github.com/michelle123lam/lloom/blob/24a7f5b1335311b90b4788608a3784a5d87e4482/text_lloom/src/text_lloom/llm.py#L34-L41).

**Example**:
```py
# Creating a LLooM instance
l = wb.lloom(
    df=df,
    text_col="your_doc_text_col",
    id_col="your_doc_id_col",  # Optional
)
```

## gen
`gen(seed=None, params=None, n_synth=1, auto_review=True)`

Runs concept generation, which includes the `Distill`-`Cluster`-`Synthesize` operator pipeline.

**Parameters**:
- `seed` _(str, optional, default: None)_: The optional seed term can steer concept induction towards more specific areas of interest (e.g., social issues" for political discussion or "evaluation methods" for academic papers). 
- `params` _(Dict, optional, default: None)_: The specific parameters to use within the Distill, Cluster, and Synthesize operators. By default, the system auto-suggests parameters based on the length and number of documents. If specified, the parameters should include the following:
    ```py
    {
        "filter_n_quotes": filter_n_quotes,  # Number of quotes per document
        "summ_n_bullets": summ_n_bullets,  # Number of bulletpoints per document
        "synth_n_concepts": synth_n_concepts,  # Number of concepts per cluster/batch
    }
    ```
- `n_synth` _(int, optional, default: 1)_: The number of times to run the Synthesize operator.
- `auto_review` _(bool, optional, default: True)_: Whether to run a step after the Synthesize operator for the system to review the set of concepts to remove low-quality concepts and merge overlapping concepts.

**Examples**:
```py
# Default version (no seed)
await l.gen()

# Or: Seed version
await l.gen(
    seed="your_optional_seed_term",
)
```

## gen_auto
`gen_auto(max_concepts=8, seed=None, params=None, n_synth=1)`

Runs concept generation, selection, and scoring as a single automated step. In `gen_auto`, the system makes a call to the LLM to automatically select which concepts to score. By contrast, `gen` only generates the concepts and allows the user to run the `select` function to review and select concepts, followed by the `score` function to perform that scoring. Returns a dataframe with the score results.

**Parameters**:
- `max_concepts` _(int, optional, default: 8)_: The maximum number of concepts for the system to select out of the set of generated concepts. All of these concepts will be scored. After concept generation, the user is prompted to confirm the set of concepts before scoring proceeds, so there is still an opportunity to review the automatic selection.
- `seed` _(str, optional, default: None)_: (Same as `gen`) The optional seed term can steer concept induction towards more specific areas of interest (e.g., social issues" for political discussion or "evaluation methods" for academic papers). 
- `params` _(Dict, optional, default: None)_: (Same as `gen`) The specific parameters to use within the Distill, Cluster, and Synthesize operators. Refer to [`gen`](#gen) for details.
- `n_synth` _(int, optional, default: 1)_: (Same as `gen`) The number of times to run the Synthesize operator.

**Example**:
```py
score_df = await l.gen_auto(
    max_concepts=5,
    seed="your_optional_seed_term", 
)
```

## select
`select()`

Allows the user to review and select concepts for scoring. Displays an interactive widget in the notebook.

**Example**:
```py
l.select()
```

In the output, each box contains the concept name, concept inclusion criteria, and representative example(s).
![LLooM select() function output](/media/ui/select_output.png)

## select_auto
`select_auto(max_concepts)`

Automatically selects up to the specified number of concepts via an LLM call.

**Parameters**:
- `max_concepts` _(int)_: The maximum number of concepts for the system to select out of the set of generated concepts. All of these concepts will be scored when the user calls the `score()` funciton.

**Example**:
```py
 await self.select_auto(max_concepts=8)
```

## show_selected
`show_selected()`

Prints out a summary of the concepts that have been currently selected. The user can make changes via the `select()` widget and re-run this function to check the current state.

**Example**:
```py
l.show_selected()
```

## score
`score(c_ids=None, batch_size=1, get_highlights=True, ignore_existing=True)`

Score all documents with respect to each concept to indicate the extent to which the document matches the concept inclusion criteria. Returns a dataframe with the score results.

**Parameters**:
- `c_ids` _(List[str], optional, default: None)_: A list of IDs (UUID strings) for the concepts that should be scored.
- `batch_size` _(int, optional, default: 1)_: Number of documents to score at once in each LLM call. By default, LLooM scores each (concept, document) combination individually to improve scoring reliability. Increasing this number will batch together multiple documents to be scored at once for a given concept.
- `get_highlights` _(bool, optional, default: True)_: Whether to retrieve highlighted quotes indicating where the document illustrates the concept.
- `ignore_existing` _(bool, optional, default: True)_: Whether to ignore concepts that have previously been scored.

**Returns**:
- `score_df` _(pd.DataFrame)_: Dataframe summarizing scoring results. Contains the following columns:
    - doc_id: Unique document ID
    - text: Text of the document
    - concept_id: Unique ID for the concept (assigned internally)
    - concept_name: Name of the concept
    - concept_prompt: Prompt conveying the concept inclusion criteria
    - score: Concept score (range: [0, 0.25, 0.5, 0.75, 1.0], where 0 indicates no concept match and 1 indicates the highest concept match)
    - rationale: LLM-provided rationale for the score
    - highlight: LLM-extracted quote from the document that illustrates the concept (if applicable) 
    - concept_seed: The seed used to generate the concept (if provided)

**Example**:
```py
score_df = await l.score()
```

## get_score_df
`get_score_df()`

Retrieves the `score_df` for the current set of active concepts.

**Returns**:
- `score_df` _(pd.DataFrame)_: Dataframe summarizing scoring results. Refer to [`score`](#score) for details.

**Example**:
```py
score_df = l.get_score_df()
```

## summary
`summary(verbose=True)`

Displays a **cumulative** summary of the (1) Total time, (2) Total cost, and (3) Tokens for the entire LLooM instance. 
- Total time: Displays the total time required for each operator. Each tuple contains the operator name and the timestamp at which the operation completed. 
- Total cost: Displays the calculated cost incurred by each operator (in US Dollars).
- Tokens: Displays the overall number of tokens used (total, in, and out)

**Parameters**:
- `verbose` _(bool, optional, default: True)_: Whether to print out verbose output (per-operator breakdowns of time and cost).

**Example**:
```py
l.summary()
```

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

## vis
`vis(cols_to_show=[], slice_col=None, max_slice_bins=5, slice_bounds=None, show_highlights=True, norm_by=None, export_df=False, include_outliers=False)`

Visualizes the concept results in the main LLooM Workbench view. An interactive widget will appear when you run this function.

**Parameters**:
- `cols_to_show` _(List[str], optional, default: [])_: Additional column names to show in the tables
- `slice_col` _(str, optional, default: None)_: Name of a column with which to slice the data. This should be a pre-existing metadata column in the dataframe. Numeric or string columns are supported. Currently, numeric columns are automatically binned into quantiles, and string columns are treated as categorical variables.
- `max_slice_bins` _(int, optional, default: 5)_: For numeric columns, the maximum number of bins to create
- `slice_bounds` _(List[float], optional, default: None)_: For numeric columns, manual bin boundaries to use. Example: `[0, 0.2, 0.4, 0.6, 0.8, 1.0]`
- `show_highlights` _(bool, optional, default: True)_: Whether to show text highlights in the table.
- `norm_by` _(str, optional, default: None)_: How to normalize scores for the matrix visualization. Options: `"concept"` or `"slice"`. If not provided, the scores will not be normalized and the matrix will reflect absolute counts.
- `export_df` _(bool, optional, default: False)_: Whether to return a dataframe for export. This dataframe contains the following columns:
    - concept: Concept name
    - criteria: Concept inclusion criteria
    - summary: Written summary of the documents that matched the concept
    - rep_examples: Representative text examples of the concept
    - prevalence: Proportion of total documents that matched this concept
    - n_matches: Absolute number of documents that matched this concept
    - highlights: Sample of highlighted text from documents that matched the concept
- `include_outliers` _(bool, optional, default: False)_: Whether to include outliers in the export_df (if requested).

**Examples**:
```py
l.vis()

# With slice column
l.vis(slice_col="n_likes")

# With normalization by concept
l.vis(slice_col="n_likes", norm_by="concept")

# With normalization by slice
l.vis(slice_col="n_likes", norm_by="slice")
```
Check out [Using the LLooM Workbench](../about/vis-guide.md) for a more detailed guide on the visualization components.
![LLooM vis() function output](/media/ui/vis_output.png)

## add
`add(name, prompt, ex_ids=[], get_highlights=True)`

Adds a new custom concepts by providing a name and prompt. This function will automatically score the data by that concept.

**Parameters**:
- `name` _(str)_: The new concept name
- `prompt` _(str)_: The new concept prompt, which conveys its inclusion criteria. 
- `ex_ids` _(List[str], optional, default: [])_: IDs of the documents that exemplify this concept.
- `get_highlights` _(bool, optional, default: True)_: Whether to retrieve highlighted quotes indicating where the document illustrates the concept.

**Example**:
```py
await l.add(
    # Your new concept name
    name="Government Critique",
    # Your new concept prompt
    prompt="Does this text criticize government actions or policies?", 
)
```

## save
`save(folder, file_name=None)`

Save the LLooM instance to a pickle file to reload at a later point.

**Parameters**:
- `folder` _(str)_: File path of the folder in which to store the pickle file.
- `file_name` _(str, optional, default: None)_: Name of the pickle file. If not specified, the system will generate a name based on the current local time.

**Example**:
```py
l.save(folder="your/path/here", file_name="your_file_name")

# Reloading later
import pickle
with open("your/path/here/your_file_name.pkl", "rb") as f:
    l = pickle.load(f)
```

## export_df
`export_df(include_outliers=False)`

Export a summary of the results in Pandas Dataframe form.

**Parameters**:
- `include_outliers` _(bool, optional, default: False)_: Whether to include the category of outliers (documents that did not match _any_ concepts) in the table.

**Returns**:
- `export_df` _(pd.DataFrame)_: Dataframe summarizing the concepts. Contains the following columns:
    - concept: Concept name
    - criteria: Concept inclusion criteria
    - summary: Written summary of the documents that matched the concept
    - rep_examples: Representative text examples of the concept
    - prevalence: Proportion of total documents that matched this concept
    - n_matches: Absolute number of documents that matched this concept
    - highlights: Sample of highlighted text from documents that matched the concept

**Example**:
```py
export_df = l.export_df()
```

## submit
`submit()`

Allows users to submit their LLooM instance to share their work, which may be selected to be featured in a gallery of results.

**Example**:
```py
l.submit()
```
You will be prompted to provide a few more details: 
- **Email address**: Please provide an email address so that we can contact you if your work is selected.
- **Analysis goal**: Share as much detail as you'd like about your analysis: What data were you using? What questions were you trying to answer? What did you find?

## estimate_gen_cost
`l.estimate_gen_cost(params=None, verbose=False)`

Estimates the cost of running `gen()` with the given parameters. The function is automatically run within calls to `gen()` for the user to review before proceeding with concept generation.

**Parameters**:
- `params` _(Dict, optional, default: None)_: The specific parameters to use within the Distill, Cluster, and Synthesize operators (see [`gen`](#gen) for details). If no parameters are provided, the function uses [auto-suggested parameters](#auto-suggest-parameters).
- `verbose` _(bool, optional, default: False)_: Whether to print a full per-operator cost breakdown

**Example**:
```py
l.estimate_gen_cost()
```

## estimate_score_cost
`estimate_score_cost(n_concepts=None, verbose=False)`

Estimates the cost of running `score()` on the provided number of concepts. The function is automatically run within calls to `score()` for the user to review before proceeding with concept scoring.

**Parameters**:
- `n_concepts` _(int, optional, default: None)_: Number of concepts to score. If not specified, the function uses the current number of active (selected) concepts.
- `verbose` _(bool, optional, default: False)_: Whether to print a full per-operator cost breakdown

**Example**:
```py
l.estimate_score_cost()
```

## auto_suggest_parameters
`auto_suggest_parameters(sample_size=None, target_n_concepts=20)`

Suggests concept generation parameters based on heuristics related to the number and length of documents in the dataset. Called automatically in `gen()` and `estimate_gen_cost()` if no parameters are provided.

**Parameters**:
- `sample_size` _(int, optional, default: None)_: Number of documents to sample from `df` to determine the parameters. If not provided, all documents will be used.
- `target_n_concepts` _(int, optional, default: 20)_: The estimated total number of concepts that the user would like to generate.

**Returns**:
- `params` _(Dict)_: The parameters to use within the Distill, Cluster, and Synthesize operators. Refer to [`gen`](#gen) for details.

**Example**:
```py
params = l.auto_suggest_parameters()
```
