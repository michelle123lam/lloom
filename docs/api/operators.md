# LLooM Operators

The LLooM Operators are a lower-level API for the operators that underlie the LLooM algorithm. The operators are defined as the `concept_induction` module within the `text_lloom` Python package. The LLooM Workbench module calls these functions internally to carry out concept induction.

```py
import text_lloom.concept_induction as ci
```

**Core operators**:
- `Distill`: Shards out and scales down data to the context window while preserving salient details.
    - `Distill-filter`: Performs extractive summarization that selects exact quotes from the original text.
    - `Distill-summarize`: Performs abstractive summarization in the form of bullet point text summaries.

- `Cluster`: Recombines shards from the Distill step into groupings that share enough meaningful overlap to induce meaningful rather than surface-level concepts

- `Synthesize`: Prompts the model to generalize from provided examples to generate concept descriptions and criteria in natural language.

- `Score`:  Labels all text documents by applying concept criteria expressed as zero- shot prompts.

**Additional operators**:
- `Seed`: Allows the user to steer concept induction. Accepts a user-provided seed term to condition the Distill or Synthesize operators, which can improve the quality and alignment of the output concepts.

- `Loop`: Further iterates on concepts by looping back to concept generation after scoring.

::: info 🚧 Under construction
Detailed documentation coming soon!
:::

## distill_filter
`distill_filter(text_df, doc_col, doc_id_col, model_name, n_quotes=3, seed=None, sess=None)`

## distill_summarize
`distill_summarize(text_df, doc_col, doc_id_col, model_name, n_bullets="2-4", n_words_per_bullet="5-8", seed=None, sess=None):`

## cluster
`cluster(text_df, doc_col, doc_id_col, cluster_id_col="cluster_id", min_cluster_size=None, embed_model_name="text-embedding-ada-002", batch_size=20, randomize=False, sess=None)`

## synthesize
`synthesize(cluster_df, doc_col, doc_id_col, model_name, cluster_id_col="cluster_id", concept_col_prefix="concept", n_concepts=None, batch_size=None, verbose=False, pattern_phrase="unifying pattern", dedupe=True, seed=None, sess=None, return_logs=False)`

## score_concepts
`score_concepts(text_df, text_col, doc_id_col, concepts, model_name="gpt-3.5-turbo", batch_size=5, get_highlights=False, sess=None, threshold=1.0)`

## loop
`loop(score_df, doc_col, doc_id_col, debug=False)`

## review
`review(concepts, concept_df, concept_col_prefix, model_name, debug=False, sess=None, return_logs=False)`