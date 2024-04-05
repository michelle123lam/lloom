# Main concept induction library functions
# =================================================

# Imports
import yaml
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import time
from tqdm.asyncio import tqdm_asyncio
import numpy as np
import math
import json
import uuid
import sys
import textwrap
from itertools import chain
import pickle
import ipywidgets as widgets

# Clustering
from bertopic import BERTopic
from bertopic.backend import OpenAIBackend
from hdbscan import HDBSCAN

# Local imports
if __package__ is None or __package__ == '':
    # uses current directory visibility
    from llm import multi_query_gpt_wrapper, calc_cost_by_tokens
    from prompts import *
    from concept import Concept
    from __init__ import MatrixWidget, ConceptSelectWidget
else:
    # uses current package visibility
    from .llm import multi_query_gpt_wrapper, calc_cost_by_tokens
    from .prompts import *
    from .concept import Concept
    from .__init__ import MatrixWidget, ConceptSelectWidget

# CONSTANTS ================================
NAN_SCORE = 0  # Numerical score to use in place of NaN values for matrix viz
OUTLIER_CRITERIA = "Did the example not match any of the above concepts?"
SCORE_DF_OUT_COLS = ["doc_id", "text", "concept_id", "concept_name", "concept_prompt", "score", "rationale", "highlight"]


# HELPER functions ================================

def json_load(s, top_level_key=None):
    # Attempts to safely load a JSON from a string response from the LLM
    if s is None:
        return None
    json_start = s.find("{")
    json_end = s.rfind("}")
    s_trimmed = s[json_start:(json_end + 1)]
    
    try:
        cur_dict = yaml.safe_load(s_trimmed)
        
        if (top_level_key is not None) and top_level_key in cur_dict:
            cur_dict = cur_dict[top_level_key]
            return cur_dict
        return cur_dict
    except:
        print(f"ERROR json_load on: {s}")
        return None

def pretty_print_dict(d):
    # Print all elements within a provided dictionary
    return "\n\t".join([f"{k}: {v}" for k, v in d.items()])

def pretty_print_dict_list(d_list):
    # Print all dictionaries in a list of dictionaries
    return "\n\t" + "\n\t".join([pretty_print_dict(d) for d in d_list])

def cluster_helper(in_df, doc_col, doc_id_col, min_cluster_size, cluster_id_col, embed_model_name):
    # OpenAI embeddings with HDBSCAN clustering
    embedding_model = OpenAIBackend(embed_model_name)

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size, 
        metric='euclidean', 
        cluster_selection_method='leaf', 
        prediction_data=True
    )
    topic_model = BERTopic(
        embedding_model=embedding_model, 
        hdbscan_model=hdbscan_model
    )

    id_vals = in_df[doc_id_col].tolist()
    text_vals = in_df[doc_col].tolist()
    clusters, probs = topic_model.fit_transform(text_vals)
    rows = list(zip(id_vals, text_vals, clusters)) # id_col, text_col, cluster_id_col
    cluster_df = pd.DataFrame(rows, columns=[doc_id_col, doc_col, cluster_id_col])
    cluster_df = cluster_df.sort_values(by=[cluster_id_col])
    
    return cluster_df

def save_progress(sess, df, step_name, start, res, model_name, debug=True):
    # Save df to session
    if (sess is not None) and (df is not None):
        k = sess.get_save_key(step_name)
        sess.saved_dfs[k] = df

    # Gets timing
    get_timing(start, step_name, sess, debug=debug)

    # Gets cost
    calc_cost(res, model_name, step_name, sess, debug=debug)

def get_timing(start, step_name, sess, debug=True):
    if start is None:
        return
    elapsed = time.time() - start
    if debug:
        print(f"Total time: {elapsed:0.2f} sec")
    if sess is not None:
        # Save to session if provided
        k = sess.get_save_key(step_name)
        sess.time[k] = elapsed

def calc_cost(results, model_name, step_name, sess, debug=True):
    # Calculate cost with API results and model name
    if results is None:
        return
    # Cost estimation
    in_tokens = np.sum([res.llm_output["token_usage"]["prompt_tokens"] for res in results])
    out_tokens = np.sum([res.llm_output["token_usage"]["completion_tokens"] for res in results])
    in_token_cost, out_token_cost = calc_cost_by_tokens(model_name, in_tokens, out_tokens)
    total_cost = in_token_cost + out_token_cost
    if debug:
        print(f"\nTotal: {total_cost} | In: {in_token_cost} | Out: {out_token_cost}")
    if sess is not None:
        # Save to session if provided
        sess.tokens["in_tokens"].append(in_tokens)
        sess.tokens["out_tokens"].append(out_tokens)
        k = sess.get_save_key(step_name)
        sess.cost[k] = total_cost

def filter_empty_rows(df, text_col_name):
    # Remove rows where the specified column is empty
    df_out = df[df[text_col_name].apply(lambda x: len(x) > 0)]
    return df_out

# CORE functions ================================

# Input: 
# - text_df: DataFrame (columns: doc_id, doc)
# Parameters: model_name, n_quotes, seed
# Output: quote_df (columns: doc_id, quote)
async def distill_filter(text_df, doc_col, doc_id_col, model_name, n_quotes=3, seed=None, sess=None):
    # Filtering operates on provided text
    start = time.time()

    # Filter to non-empty rows
    text_df = filter_empty_rows(text_df, doc_col)

    # Prepare prompts
    filtered_ex = []
    rows = []
    if seed is not None:
        seeding_phrase = f"related to {seed}"
    else:
        seeding_phrase = "most important"
    arg_dicts = [
        {
            "ex": ex, 
            "n_quotes": n_quotes, 
            "seeding_phrase": seeding_phrase.upper()
        } for ex in text_df[doc_col].tolist()
    ]
    
    # Run prompts
    prompt_template = filter_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name)

    # Process results
    ex_ids = [ex_id for ex_id in text_df[doc_id_col].tolist()]
    for ex_id, res in zip(ex_ids, res_text):
        cur_filtered_list = json_load(res, top_level_key="relevant_quotes")
        if cur_filtered_list is not None:
            cur_filtered = "\n".join(cur_filtered_list)
            filtered_ex.append(cur_filtered)
            rows.append([ex_id, cur_filtered])
    quote_df = pd.DataFrame(rows, columns=[doc_id_col, doc_col])
    
    save_progress(sess, quote_df, step_name="distill_filter", start=start, res=res_full, model_name=model_name)
    return quote_df


# Input: text_df (columns: doc_id, doc) 
#   --> text could be original or filtered (quotes)
# Parameters: n_bullets, n_words_per_bullet, seed
# Output: bullet_df (columns: doc_id, bullet)
async def distill_summarize(text_df, doc_col, doc_id_col, model_name, n_bullets="2-4", n_words_per_bullet="5-8", seed=None, sess=None):
    # Summarization operates on text_col
    start = time.time()

    # Filter to non-empty rows
    text_df = filter_empty_rows(text_df, doc_col)

    # Prepare prompts
    rows = []
    arg_dicts = []
    all_ex_ids = []

    if seed is not None:
        seeding_phrase = f"related to {seed}"
    else:
        seeding_phrase = ""
    for _, row in text_df.iterrows():
        ex = row[doc_col]
        ex_id = row[doc_id_col]
        if len(ex) == 0:
            # Handle if filtered example is empty
            rows.append([ex_id, ""])
            continue
        arg_dict = {
            "ex": ex, 
            "seeding_phrase": seeding_phrase.upper(), 
            "n_bullets": n_bullets, 
            "n_words": n_words_per_bullet
        }
        arg_dicts.append(arg_dict)
        all_ex_ids.append(ex_id)
    
    # Run prompts
    prompt_template = summarize_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name)

    # Process results
    for ex_id, res in zip(all_ex_ids, res_text):
        cur_bullets_list = json_load(res, top_level_key="bullets")
        if cur_bullets_list is not None:
            for bullet in cur_bullets_list:
                # Expand each bullet into its own row
                rows.append([ex_id, bullet])
    bullet_df = pd.DataFrame(rows, columns=[doc_id_col, doc_col])

    save_progress(sess, bullet_df, step_name="distill_summarize", start=start, res=res_full, model_name=model_name)
    return bullet_df


# Input: text_df (columns: doc_id, doc) 
#   --> text could be original, filtered (quotes), and/or summarized (bullets)
# Parameters: n_clusters
# Output: cluster_df (columns: doc_id, doc, cluster_id)
async def cluster(text_df, doc_col, doc_id_col, cluster_id_col="cluster_id", min_cluster_size=None, embed_model_name="text-embedding-ada-002", batch_size=20, randomize=False, sess=None):
    # Clustering operates on text_col
    start = time.time()

    # Filter to non-empty rows
    text_df = filter_empty_rows(text_df, doc_col)

    # Auto-set parameters
    n_items = len(text_df)
    if min_cluster_size is None:
        min_cluster_size = max(3, int(n_items/10))
    n_batches = math.ceil(n_items / batch_size)

    # Generate clusters
    if randomize:
        # Randomize the bulletpoints
        text_df = text_df.sample(frac=1)  # shuffle order
        cluster_df = text_df.copy()
        cluster_ids = [
            [i for _ in range(batch_size)] 
            for i in range(n_batches)
        ]
        cluster_ids = list(chain.from_iterable(cluster_ids))[:n_items]
        cluster_df[cluster_id_col] = cluster_ids
    else:
        # Cluster and group by clusters
        cluster_df = cluster_helper(text_df, doc_col, doc_id_col, min_cluster_size=min_cluster_size, cluster_id_col=cluster_id_col, embed_model_name=embed_model_name)

    save_progress(sess, cluster_df, step_name="cluster", start=start, res=None, model_name=None)
    return cluster_df


def dict_to_json(examples):
    # Internal helper to convert examples to json for prompt
    examples_json = json.dumps(examples)
    # Escape curly braces to avoid the system interpreting as template formatting
    examples_json = examples_json.replace("{", "{{")
    examples_json = examples_json.replace("}", "}}")
    return examples_json

# Input: cluster_df (columns: doc_id, doc, cluster_id)
# Parameters: n_concepts
# Output: 
# - concepts: dict (concept_id -> concept dict)
# - concept_df: DataFrame (columns: doc_id, doc, concept_id, concept_name, concept_prompt)
async def synthesize(cluster_df, doc_col, doc_id_col, model_name, cluster_id_col="cluster_id", concept_col_prefix="concept", n_concepts=None, batch_size=None, verbose=True, pattern_phrase="unifying pattern", dedupe=True, seed=None, sess=None):
    # Synthesis operates on "doc" column for each cluster_id
    # Concept object is created for each concept
    start = time.time()
    
    # Filter to non-empty rows
    cluster_df = filter_empty_rows(cluster_df, doc_col)

    # Auto-set parameters
    def get_n_concepts_phrase(cur_set):
        if n_concepts is None:
            cur_n_concepts = math.ceil(len(cur_set)/3)
        else:
            cur_n_concepts = n_concepts
        if cur_n_concepts > 1:
            return f"up to {cur_n_concepts} {pattern_phrase}s"
        else:
            return f"{cur_n_concepts} {pattern_phrase}"
    
    # Prepare prompts
    # Create prompt arg dictionary with example IDs
    if seed is not None:
        seed_phrase = f"If possible, please make the patterns RELATED TO {seed.upper()}."
    else:
        seed_phrase = ""
    arg_dicts = []
    cluster_ids = cluster_df[cluster_id_col].unique()
    cluster_dfs = {}  # Store each cluster's dataframe by cluster_id
    ex_id_to_ex = {(str(row[doc_id_col]), row[cluster_id_col]): row[doc_col] for _, row in cluster_df.iterrows()}  # Map example IDs to example text
    for cluster_id in cluster_ids:
        # Iterate over cluster IDs to get example sets
        cur_df = cluster_df[cluster_df[cluster_id_col] == cluster_id]
        cluster_dfs[cluster_id] = cur_df
        if batch_size is not None:
            # Split into batches
            n_batches = math.ceil(len(cur_df) / batch_size)
            for i in range(n_batches):
                cur_batch_df = cur_df.iloc[i*batch_size:(i+1)*batch_size]
                ex_dicts = [{"example_id": row[doc_id_col], "example": row[doc_col]} for _, row in cur_batch_df.iterrows()]
                ex_dicts_json = dict_to_json(ex_dicts)
                arg_dict = {
                    "examples": ex_dicts_json,
                    "n_concepts_phrase": get_n_concepts_phrase(cur_df),
                    "seed_phrase": seed_phrase
                }
                arg_dicts.append(arg_dict)
        else:
            # Handle unbatched case
            ex_dicts = [{"example_id": row[doc_id_col], "example": row[doc_col]} for _, row in cur_df.iterrows()]
            ex_dicts_json = dict_to_json(ex_dicts)
            arg_dict = {
                "examples": ex_dicts_json,
                "n_concepts_phrase": get_n_concepts_phrase(cur_df),
                "seed_phrase": seed_phrase
            }
            arg_dicts.append(arg_dict)

    # Run prompts
    prompt_template = synthesize_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name)

    # Process results
    concepts = {}
    rows = []
    for cur_cluster_id, res in zip(cluster_ids, res_text):
        cur_concepts = json_load(res, top_level_key="patterns")
        if cur_concepts is not None:
            # concepts.extend(cur_concepts)
            # cur_examples_dict = {ex_to_id[cur_ex]: cur_ex for cur_ex in cur_examples}
            
            for concept_dict in cur_concepts:
                ex_ids = concept_dict["example_ids"]
                ex_ids = set(ex_ids) # remove duplicates
                concept = Concept(
                    name=concept_dict["name"],
                    prompt=concept_dict["prompt"],
                    example_ids=ex_ids,
                    active=False,
                )
                concepts[concept.id] = concept
                
                for ex_id in ex_ids:
                    # doc_id, text, concept_id, concept_name, concept_prompt
                    cur_key = (ex_id, cur_cluster_id)
                    if cur_key in ex_id_to_ex:
                        row = [ex_id, ex_id_to_ex[cur_key], concept.id, concept.name, concept.prompt]
                        rows.append(row)
            if verbose:
                examples = cluster_dfs[cur_cluster_id][doc_col].tolist()
                concepts_formatted = pretty_print_dict_list(cur_concepts)
                print(f"\n\nInput examples: {examples}\nOutput concepts: {concepts_formatted}")
    # doc_id, text, concept_id, concept_name, concept_prompt
    concept_df = pd.DataFrame(rows, columns=[doc_id_col, doc_col, concept_col_prefix, f"{concept_col_prefix}_name", f"{concept_col_prefix}_prompt"])

    concept_df[f"{concept_col_prefix}_namePrompt"] = concept_df[f"{concept_col_prefix}_name"] + ": " + concept_df[f"{concept_col_prefix}_prompt"]
    if dedupe:
        concept_df = dedupe_concepts(concept_df, concept_col=f"{concept_col_prefix}_namePrompt")

    save_progress(sess, concept_df, step_name="synthesize", start=start, res=res_full, model_name=model_name)
    # Save to session if provided
    if sess is not None:
        for c_id, c in concepts.items():
            sess.concepts[c_id] = c
    return concept_df

def dedupe_concepts(df, concept_col):
    # Remove duplicate concept rows
    return df.drop_duplicates(subset=[concept_col])

def pretty_print_merge_results(merged):
    for m in merged:
        orig_concepts = m["original_themes"]
        print(f"[{orig_concepts}] --> {m['merged_theme_name']}: {m['merged_theme_prompt']}")

# Input: concept_df (columns: doc_id, text, concept_id, concept_name, concept_prompt)
# Parameters: n_concepts
# Output: 
# - concepts: dict (concept_id -> concept dict)
# - concept_df: DataFrame (columns: doc_id, text, concept_id, concept_name, concept_prompt)
async def review(concepts, concept_df, concept_col, concept_col_prefix, model_name, debug=True, sess=None):
    # Model is asked to review the provided set of concepts
    concepts_out, concept_df_out, removed = await review_remove(concepts, concept_df, concept_col, concept_col_prefix, model_name=model_name, sess=sess)
    concepts_out, concept_df_out, merged = await review_merge(concepts_out, concept_df_out, concept_col, concept_col_prefix, model_name=model_name, sess=sess)

    if debug:
        print(f"Removed ({len(removed)}):\n{removed}")
        print(f"Merged ({len(merged)}):")
        pretty_print_merge_results(merged)

    # TODO: ask model to filter to the "best" N concepts
    return concepts_out, concept_df_out


# Model removes concepts that are too specific or too general
# Input: concept_df (columns: doc_id, text, concept_id, concept_name, concept_prompt)
# Parameters: n_concepts
# Output: 
# - concepts: dict (concept_id -> concept dict)
# - concept_df: DataFrame (columns: doc_id, text, concept_id, concept_name, concept_prompt)
async def review_remove(concepts, concept_df, concept_col, concept_col_prefix, model_name, sess):
    concepts = concepts.copy()  # Make a copy of the concepts dict to avoid modifying the original
    start = time.time()
    concept_name_col = f"{concept_col_prefix}_name"

    concepts_list = concept_df[concept_col].tolist()
    concepts_list = [f"- {c}" for c in concepts_list]
    concepts_list_str = "\n".join(concepts_list)
    arg_dicts = [{
        "themes": concepts_list_str,
    }]

    # Run prompts
    prompt_template = review_remove_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name)

    # Process results
    res = res_text[0]
    concepts_to_remove = json_load(res, top_level_key="remove")

    concept_df_out = concept_df.copy()
    concept_df_out = concept_df_out[~concept_df_out[concept_name_col].isin(concepts_to_remove)]
    c_ids_to_remove = []
    for c_id, c in concepts.items():
        if c['name'] in concepts_to_remove:
            c_ids_to_remove.append(c_id)
    for c_id in c_ids_to_remove:
        concepts.pop(c_id, None)

    save_progress(sess, concept_df_out, step_name="review_remove", start=start, res=res_full, model_name=model_name)
    return concepts, concept_df_out, concepts_to_remove

def get_concept_by_name(concepts, concept_name):
    for c_id, c in concepts.items():
        if c['name'] == concept_name:
            return c_id, c
    return None, None

# Model merges concepts that are similar or overlapping
# Input: concept_df (columns: doc_id, text, concept_id, concept_name, concept_prompt)
# Parameters: n_concepts
# Output: 
# - concepts: dict (concept_id -> concept dict)
# - concept_df: DataFrame (columns: doc_id, text, concept_id, concept_name, concept_prompt)
async def review_merge(concepts, concept_df, concept_col, concept_col_prefix, model_name, sess):
    concepts = concepts.copy()  # Make a copy of the concepts dict to avoid modifying the original
    start = time.time()
    concept_name_col = f"{concept_col_prefix}_name"
    concept_prompt_col = f"{concept_col_prefix}_prompt"

    concepts_list = concept_df[concept_col].tolist()
    concepts_list = [f"- {c}" for c in concepts_list]
    concepts_list_str = "\n".join(concepts_list)
    arg_dicts = [{
        "themes": concepts_list_str,
    }]

    # Run prompts
    prompt_template = review_merge_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name)

    # Process results
    res = res_text[0]
    concepts_to_merge = json_load(res, top_level_key="merge")

    concept_df_out = concept_df.copy()

    # Remove all original concepts
    # Add new merged concepts
    concepts_to_remove = []
    c_ids_to_remove = []
    for merge_result in concepts_to_merge:
        orig_concepts = merge_result["original_themes"]
        # Only allow merging pairs
        if len(orig_concepts) != 2:
            continue
        
        # Don't allow duplicates (in prior merge pairs)
        for orig_concept in orig_concepts:
            if orig_concept in concepts_to_remove:
                continue

        concepts_to_remove.extend(orig_concepts)

        # Get original concept IDs and example IDs
        merged_example_ids = []
        for orig_concept in orig_concepts:
            c_id, c = get_concept_by_name(concepts, orig_concept)
            if c is not None:
                c_ids_to_remove.append(c_id)
                merged_example_ids.extend(c["example_ids"])
        
        # Create new merged concept in dict
        new_concept_name = merge_result["merged_theme_name"]
        new_concept_prompt = merge_result["merged_theme_prompt"]
        new_concept_id = str(uuid.uuid4())
        concepts[new_concept_id] = {
            "name": new_concept_name,
            "prompt": new_concept_prompt,
            "example_ids": merged_example_ids,
        }

        # Replace prior df row with new merged concept
        for orig_concept in orig_concepts:
            concept_df_out.loc[concept_df_out[concept_name_col]==orig_concept, concept_name_col] = new_concept_name
            concept_df_out.loc[concept_df_out[concept_name_col]==orig_concept, concept_prompt_col] = new_concept_prompt
            concept_df_out.loc[concept_df_out[concept_name_col]==orig_concept, concept_col] = f"{new_concept_name}: {new_concept_prompt}"
        
    for c_id in c_ids_to_remove:
        concepts.pop(c_id, None)

    save_progress(sess, concept_df_out, step_name="review_merge", start=start, res=res_full, model_name=model_name)
    return concepts, concept_df_out, concepts_to_merge


def get_ex_batch_args(df, text_col, doc_id_col, concept_name, concept_prompt):
    ex = get_examples_dict(df, text_col, doc_id_col)
    examples_json = dict_to_json(ex)
    arg_dict = {
        "examples_json": examples_json,
        "pattern_name": concept_name,
        "pattern_prompt": concept_prompt,
        "example_ids": list(df[doc_id_col])
    }
    return arg_dict

def get_examples_dict(cur_df, text_col, doc_id_col):
    # Helper to get examples from cur_df in dictionary form for JSON in prompt
    ex_list = []
    for i, row in cur_df.iterrows():
        ex_dict = {
            "example_id": row[doc_id_col],
            "example_text": row[text_col],
        }
        ex_list.append(ex_dict)

    ex = {"cur_examples": ex_list}
    return ex

def parse_bucketed_score(x):
    # Internal helper to parse bucketed score from LLM response to numerical result
    answer_scores = {
        "A": 1.0,
        "B": 0.75,
        "C": 0.5,
        "D": 0.25,
        "E": 0.0,
    }
    if len(x) > 1:
        x = x[0]
    if x not in answer_scores.keys():
        return NAN_SCORE
    return answer_scores[x]

def get_score_df(res, in_df, concept, concept_id, text_col, doc_id_col, get_highlights):
    # Cols: doc_id, text, concept_id, concept_name, concept_prompt, score, highlight
    res_dict = json_load(res, top_level_key="pattern_results")
    concept_name = concept.name
    concept_prompt = concept.prompt
    if res_dict is not None:
        rows = []
        for ex in res_dict:
            if "answer" in ex:
                ans = parse_bucketed_score(ex["answer"])
            else:
                ans = NAN_SCORE
            
            if "example_id" in ex:
                doc_id = ex["example_id"]
                text_list = in_df[in_df[doc_id_col] == doc_id][text_col].tolist()
                if len(text_list) > 0:
                    # Document is in the dataset
                    text = text_list[0]

                    if "rationale" in ex:
                        rationale = ex["rationale"]
                    else:
                        rationale = ""   # Set rationale to empty string

                    if get_highlights and ("quote" in ex):
                        row = [doc_id, text, concept_id, concept_name, concept_prompt, ans, rationale, ex["quote"]]
                    else:
                        row = [doc_id, text, concept_id, concept_name, concept_prompt, ans, rationale, ""]  # Set quote to empty string
                    rows.append(row)
        
        out_df = pd.DataFrame(rows, columns=SCORE_DF_OUT_COLS)
        return out_df
    else:
        out_df = get_empty_score_df(in_df, concept, concept_id, text_col, doc_id_col)
        return out_df[SCORE_DF_OUT_COLS]

def get_empty_score_df(in_df, concept, concept_id, text_col, doc_id_col):
    # Cols: doc_id, text, concept_id, concept_name, concept_prompt, score, highlight
    concept_name = concept.name
    concept_prompt = concept.prompt
    out_df = in_df.copy()
    out_df["doc_id"] = out_df[doc_id_col]
    out_df["text"] = out_df[text_col]
    out_df["concept_id"] = concept_id
    out_df["concept_name"] = concept_name
    out_df["concept_prompt"] = concept_prompt
    out_df["score"] = NAN_SCORE
    out_df["rationale"] = ""
    out_df["highlight"] = ""
    return out_df[SCORE_DF_OUT_COLS]

# Performs scoring for one concept
async def score_helper(concept, batch_i, concept_id, df, text_col, doc_id_col, model_name, batch_size, get_highlights, sess):
    # TODO: add support for only a sample of examples
    # TODO: set consistent concept IDs for reference

    # Prepare batches of input arguments
    indices = range(0, len(df), batch_size)
    ex_ids = [str(x) for x in df[doc_id_col].tolist()]
    ex_id_sets = [ex_ids[i:i+batch_size] for i in indices]
    in_dfs = [df[df[doc_id_col].isin(cur_ex_ids)] for cur_ex_ids in ex_id_sets]
    arg_dicts = [
        get_ex_batch_args(df, text_col, doc_id_col, concept.name, concept.prompt) for df in in_dfs
    ]

    # Run prompts in parallel to score each example
    if get_highlights:
        prompt_template = score_highlight_prompt
    else:
        prompt_template = score_no_highlight_prompt
    results, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name, batch_num=batch_i)

    # Parse results
    # Cols: doc_id, text, concept_id, concept_name, concept_prompt, score, highlight
    score_df = None
    for i, res in enumerate(results):
        in_df = in_dfs[i]
        cur_score_df = get_score_df(res, in_df, concept, concept_id, text_col, doc_id_col, get_highlights)
        if score_df is None:
            score_df = cur_score_df
        else:
            score_df = pd.concat([score_df, cur_score_df])

    # Fill in missing rows if necessary
    # TODO: Add automatic retries in case of missing rows
    if len(score_df) < len(df):
        missing_rows = df[~df[doc_id_col].isin(score_df["doc_id"])]
        missing_rows = get_empty_score_df(missing_rows, concept, concept_id, text_col, doc_id_col)
        score_df = pd.concat([score_df, missing_rows])

    save_progress(sess, score_df, step_name="score_helper", start=None, res=res_full, model_name=model_name)
    if sess is not None:
        # Save to session if provided
        sess.results[concept_id] = score_df
    
        # Generate summary
        cur_summary = await summarize_concept(score_df, concept_id, model_name, sess=sess)
    return score_df

# Performs scoring for all concepts
# Input: concepts, text_df (columns: doc_id, text)
#   --> text could be original, filtered (quotes), and/or summarized (bullets)
# Parameters: threshold
# Output: score_df (columns: doc_id, text, concept_id, concept_name, concept_prompt, score, highlight)
async def score_concepts(text_df, text_col, doc_id_col, concepts, model_name="gpt-3.5-turbo", batch_size=5, get_highlights=False, sess=None):
    # Scoring operates on "text" column for each concept
    start = time.time()

    text_df = text_df.copy()
    # Filter to non-empty rows
    text_df = filter_empty_rows(text_df, text_col)

    text_df[doc_id_col] = text_df[doc_id_col].astype(str)
    tasks = [score_helper(concept, concept_i, concept_id, text_df, text_col, doc_id_col, model_name, batch_size, get_highlights, sess=sess) for concept_i, (concept_id, concept) in enumerate(concepts.items())]
    score_dfs = await tqdm_asyncio.gather(*tasks, file=sys.stdout)

    # Combine score_dfs
    score_df = pd.concat(score_dfs)

    save_progress(sess, score_df, step_name="score", start=start, res=None, model_name=None)
    return score_df

# Based on concept scoring, refine concepts
# Input: score_df (columns: doc_id, text, concept_id, concept_name, concept_prompt, score, highlight), concepts
# Parameters: 
# - threshold (float): minimum score of positive class
# - generic_threshold (float): min fraction concept matches to be considered generic
# - rare_threshold (float): max fraction of concept matches considered rare
# Output: 
# - concepts (dict)
def refine(score_df, concepts, threshold=1, generic_threshold=0.75, rare_threshold=0.05, debug=True):
    # Check for generic, rare, and redundant concepts
    # TODO: add support for redundant concepts
    concepts = concepts.copy()  # Make a copy of the concepts dict to avoid modifying the original
    generic = []
    rare = []
    concept_ids = score_df["concept_id"].unique().tolist()
    for c_id in concept_ids:
        cur_concept_df = score_df[score_df["concept_id"] == c_id]
        cur_concept_pos = score_df[(score_df["concept_id"] == c_id) & (score_df["score"] >= threshold)]
        # Get fraction of positive labels out of all examples
        pos_frac = len(cur_concept_pos) / len(cur_concept_df)
        if pos_frac >= generic_threshold:
            if debug:
                print(f"Generic: {concepts[c_id]['name']}, {pos_frac} match")
            generic.append(c_id)
        elif pos_frac < rare_threshold:
            if debug:
                print(f"Rare: {concepts[c_id]['name']}, {pos_frac} match")
            rare.append(c_id)

    # Remove identified concepts
    if debug:
        print(f"Generic ({len(generic)}): {[concepts[c_id]['name'] for c_id in generic]}")
        print(f"Rare ({len(rare)}): {[concepts[c_id]['name'] for c_id in rare]}")
    for c_id in generic:
        concepts.pop(c_id, None)
    for c_id in rare:
        concepts.pop(c_id, None)
    
    return concepts

async def summarize_concept(score_df, concept_id, model_name="gpt-4-turbo-preview", sess=None, threshold=0.75, summary_length="15-20 word", score_col="score", highlight_col="highlight"):
    # Summarizes behavior in each concept
    df = score_df.copy()
    df = df[df[score_col] >= threshold]

    # Prepare inputs
    arg_dicts = []
    cur_df = df[df["concept_id"] == concept_id]
    cur_df = cur_df.sample(frac=1)  # shuffle order
    if len(cur_df) == 0:
        # No concept matches to summarize
        return None
    concept_name = cur_df["concept_name"].iloc[0]
    concept_prompt = cur_df["concept_prompt"].iloc[0]

    arg_dicts = [{
        "concept_name": concept_name,
        "concept_prompt": concept_prompt,
        "examples": cur_df[highlight_col].tolist(),
        "summary_length": summary_length
    }]
    
    # Run prompts
    prompt_template = summarize_concept_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name)

    # Process results
    res = res_text[0]
    cur_summary = json_load(res, top_level_key="summary")
    if cur_summary is not None:
        if sess is not None:
            sess.concepts[concept_id].summary = cur_summary
        else:
            return cur_summary
    return None

# Returns ids of not-covered documents
def get_not_covered(score_df, doc_id_col, threshold=0.75, debug=False):
    # Convert to thresholded scores; sum scores across concepts for each doc
    df = score_df.copy()
    df["score"] = df["score"].apply(lambda x: 1 if x >= threshold else 0)
    df = df.groupby(doc_id_col).sum().reset_index() # Sum scores for each doc
    if debug:
        display(df)

    # Filter to examples with no positive scores
    df = df[df["score"] == 0]
    doc_ids = df[doc_id_col].tolist()
    if debug:
        print(doc_ids)
    return doc_ids

# Returns ids of covered-by-generic documents
def get_covered_by_generic(score_df, doc_id_col, threshold=0.75, generic_threshold=0.5, debug=False):
    # Determines generic concepts
    df = score_df.copy()
    df["score"] = df["score"].apply(lambda x: 1 if x >= threshold else 0)
    df_generic = df.groupby("concept_id").mean().reset_index()
    df_generic.rename(columns={"score": "pos_frac"}, inplace=True)
    df_generic = df_generic[df_generic["pos_frac"] >= generic_threshold]
    generic_concepts = df_generic["concept_id"].unique().tolist()
    if debug:
        display(df_generic)
        print(generic_concepts)

    # Determines covered-by-generic documents (those that only match generic concepts)
    # Remove rows for generic concepts and filter to examples with no positive scores
    df_out = score_df.copy()
    df_out = df_out[~df_out["concept_id"].isin(generic_concepts)]
    df_out = df_out.groupby(doc_id_col).sum().reset_index() # Sum scores for each doc
    df_out = df_out[df_out["score"] == 0]
    doc_ids = df_out[doc_id_col].tolist()
    if debug:
        print(doc_ids)
    return doc_ids

# Input: score_df (columns: doc_id, doc, concept_id, score)
# Output: text_df (columns: doc_id, doc) --> filtered set of documents to run further iterations of concept induction
# Returns None if (1) there are zero remaining documents after filtering or (2) there are no changes in the number of documents after filtering
def loop(score_df, doc_col, doc_id_col, debug=False):
    # Check for not-covered and covered-by-generic documents
    # Not-covered: examples that don't match any concepts
    # Covered-by-generic: examples that only match generic concepts (those which cover the majority of examples)
    # TODO: Allow users to decide on custom filtering conditions
    # TODO: Save generic concepts to session (to avoid later)
    n_initial = len(score_df[doc_id_col].unique())

    underrep_ids = get_not_covered(score_df, doc_id_col)
    generic_ids = get_covered_by_generic(score_df, doc_id_col)
    ids_to_include = underrep_ids + generic_ids
    ids_to_include = set(ids_to_include)
    if debug:
        print(f"ids_to_include ({len(ids_to_include)}): {ids_to_include}")

    text_df = score_df.copy()
    text_df = text_df[text_df[doc_id_col].isin(ids_to_include)][[doc_id_col, doc_col]].drop_duplicates().reset_index()

    # Stopping condition: if text_df is empty or has the same number of docs as the original df
    n_final = len(text_df[doc_id_col].unique())
    if (n_final == n_initial) or (len(text_df) == 0):
        return None
    return text_df


def trace():
    # Input: concept_df (columns: doc_id, text, concept_id, ...), text_dfs (columns: doc_id, text)
    # Output: trace_df (columns: doc_id, text, concept_id, score, text1, text2)

    # Joins the score_df with other text dfs that share the same doc_id column
    # To trace from the final concepts and scores to the original text
    pass

def parse_tf_answer(ans):
    if ans.lower() == "true":
        return True
    elif ans.lower() == "false":
        return False
    return False

def clean_item_id(x):
    x = x.replace("item_id ", "")
    x = x.strip()
    return x
    
# Eval helpers
# Input: items (dict), concepts (list of strings)
# Output: concepts_found (dict: concept_id -> list of items), concept_coverage (float)
async def auto_eval(items, concepts, model_name="gpt-3.5-turbo", debug=False, sess=None):
    # Iterate through all concepts to check whether they match any of the items
    start = time.time()

    items_str = "\n".join([f"- item_id {item_id}: {item['name']}. {item['prompt']}" for item_id, item in items.items()])
    items_dict = {str(item_id): item for item_id, item in items.items()}
    item_names_dict = {item['name']: item for _, item in items.items()}
    concepts_dict = {str(i): c for i, c in enumerate(concepts)}
    concepts_str = "\n".join([f"- concept_id {c_id}: {concept}" for c_id, concept in concepts_dict.items()])

    arg_dicts = [
        {
            "concepts": concepts_str,
            "items": items_str,
        }
    ]

    if debug:
        print(arg_dicts)

    # Run prompts
    prompt_template = concept_auto_eval_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name)
    
    res_text = res_text[0]
    if debug:
        print(res_text)

    # Return the concept coverage and mapping from concepts to items
    concepts_found = {}
    concepts_found_unique = {}
    items_added = []
    matches = json_load(res_text, top_level_key="concept_matches")
    if matches is not None:
        for match in matches:
            concept_id = str(match["concept_id"])
            item_id = str(match["item_id"])
            if "item_id" in item_id:
                item_id = clean_item_id(item_id)
            if (concept_id not in concepts_dict) or (item_id not in items_dict):
                continue
            item = items_dict[item_id]
            concepts_found[concept_id] = item
            if item not in items_added:
                concepts_found_unique[concept_id] = item
                items_added.append(item)
    
    if (len(concepts_found) == 0) and (matches is not None):
        # Check for case where items were provided in plain text
        for match in matches:
            concept_id = str(match["concept_id"])
            item_name = str(match["item_id"])
            if (concept_id not in concepts_dict) or (item_name not in item_names_dict):
                continue
            item = item_names_dict[item_name]
            concepts_found[concept_id] = item
            if item not in items_added:
                concepts_found_unique[concept_id] = item
                items_added.append(item)
    
    # Display metrics
    concept_coverage = len(concepts_found_unique) / len(concepts)
    print(f"Concept coverage: {np.round(concept_coverage, 2)}")

    print("Concepts found:")
    for c_id, c in concepts_dict.items():
        if c_id in concepts_found:
            print(f"{c_id}: {c} -- {concepts_found[c_id]['name']}: {concepts_found[c_id]['prompt']}")
        else:
            print(f"{c_id}: {c} -- NONE")

    save_progress(sess, df=None, step_name="auto_eval", start=start, res=res_full, model_name=model_name)
    return concepts_found, concept_coverage

# Adds color-background styling for score columns to match score value
def format_scores(x: float):
    color_str = f"rgba(130, 193, 251, {x*0.5})"
    start_tag = f"<div class='score-col' style='background-color: {color_str}'>"
    return f"{start_tag}{x}</div>"

# Converts a list of bullet strings to a formatted unordered list
def format_bullets(orig, add_quotes=False):
    if (not isinstance(orig, list)) or len(orig) == 0:
        return ""
    if add_quotes:
        lines = [f"<li>\"{line}\"</li>" for line in orig]
    else:
        lines = [f"<li>{line}</li>" for line in orig]
    return "<ul>" + "".join(lines) + "</ul>"

# Adds color-background styling for highlight columns to match score value
def format_highlight(orig: str, quotes, x: float):
        color_str = f"rgba(130, 193, 251, {x*0.5})"
        start_tag = f"<span style='background-color: {color_str}'>"
        end_tag = "</span>"
        if (not isinstance(quotes, str)):
            # Skip if quotes are not a valid string
            return orig
        quotes = quotes.split("\n")
        for quote in quotes:
            quote = str(quote)  # Cast result to string
            orig = str(orig)
            if quote in orig:
                orig = orig.replace(quote, f"{start_tag}{quote}{end_tag}")
        return orig

# Transforms scores to be between 0 and 1
def clean_score(x, threshold):
    if pd.isna(x):
        return 0.0
    elif (x is True) or (x == "True"):
        return 1.0
    elif (x is False) or (x == "False"):
        return 0.0
    else:
        if threshold is None:
            return x
        if x < threshold:
            return 0.0
        return 1.0

# Check if the given example (represented by a row) is an outlier (i.e., doesn't match any of the concepts)
def is_outlier(row, concept_names, threshold):
    concept_matches = [clean_score(row[c], threshold) for c in concept_names]
    return not np.any(concept_matches)

# Helper function for `prep_vis_dfs()` to generate a dataframe with a column containing the scores for each concept
# The output dataframe has: doc_id_col, doc_col, any specified cols_to_show, and a column for each concept (where the column name is the concept name)
def get_concept_col_df(df, score_df, concepts, doc_id_col, doc_col, score_col, cols_to_show):
    id_cols = [doc_id_col, doc_col] + cols_to_show
    cur_df = df[id_cols].copy()
    concept_cols_to_show = [doc_id_col, score_col]
    for c_id, c in concepts.items():
        c_df = score_df[score_df["concept_id"] == c_id][concept_cols_to_show]
        c_df = c_df.rename(columns={score_col: c.name}) # Store score under concept name
        # Rename columns and remove unused columns
        cur_df[doc_id_col] = cur_df[doc_id_col].astype(str)
        c_df[doc_id_col] = c_df[doc_id_col].astype(str)
        cur_df = cur_df.merge(c_df, on=doc_id_col, how="left")
    return cur_df

# Template slice function for string (categorical) columns
def _slice_fn_cat(x, group_name):
    return x == group_name

# Template slice function for numeric columns
def _slice_fn_num(x, left, right):
    return (x > left) and (x <= right)

# Convert manual slice bounds to bins
def slice_bounds_to_bins(slice_bounds):
    bins = []
    for i in range(len(slice_bounds) - 1):
        bins.append(pd.Interval(left=slice_bounds[i], right=slice_bounds[i+1]))
    return bins

# Automatically create slice groupings based on slice column
# - slice_col: str (column name with which to slice data)
# - max_slice_bins: int (Optional: for numeric columns, the maximum number of bins to create)
# - slice_bounds: list (Optional: for numeric columns, manual bin boundaries to use)
def get_groupings(df, slice_col, max_slice_bins, slice_bounds):
    # Determine type to create groupings
    if is_numeric_dtype(df[slice_col]):
        # Numeric column: Create bins
        if slice_bounds is not None:
            # Use provided bin boundaries
            bins = slice_bounds_to_bins(slice_bounds)
        else:
            # Automatically create bins using percentiles
            bin_assn = pd.qcut(df[slice_col], q=max_slice_bins, duplicates="drop", labels=None)
            bins = sorted(bin_assn.unique(), key=lambda x: x.left if (isinstance(x, pd.Interval)) else 0, reverse=False)
        def get_bin_name(bin):
            if isinstance(bin, pd.Interval):
                return f"({bin.left}, {bin.right}]"
            return f"{bin}"
        def get_bin_fn(bin):
            if isinstance(bin, pd.Interval):
                return {"x": slice_col, "fn": _slice_fn_num, "args": [bin.left, bin.right]}
            elif pd.isna(bin):
                return {"x": slice_col, "fn": lambda x: pd.isna(x), "args": []}
            return {"x": slice_col, "fn": _slice_fn_cat, "args": [bin]}
        groupings = {
            get_bin_name(bin): get_bin_fn(bin) for bin in bins
        }
    elif is_string_dtype(df[slice_col]):
        # String column: Create groupings based on unique values
        def get_group_name(group_name):
            return f"{group_name}"
        groupings = {
            get_group_name(group_name): {"x": slice_col, "fn": _slice_fn_cat, "args": [group_name]} for group_name in df[slice_col].unique()
        }
    else:
        raise ValueError(f"Slice column type not supported: {df[slice_col].dtype}. Please convert this column to numeric or string type.")
        groupings = {}
    return groupings

# Helper function for `visualize()` to generate the underlying dataframes
# Parameters:
# - threshold: float (minimum score of positive class)
def prep_vis_dfs(df, score_df, doc_id_col, doc_col, score_col, df_filtered, df_bullets, concepts, cols_to_show, slice_col, max_slice_bins, slice_bounds, show_highlights, norm_by=None, debug=False, threshold=None, outlier_threshold=0.75):
    # TODO: codebook info

    # Handle groupings
    # Add the "All" grouping by default
    groupings = {
        "All": {"x": None, "fn": None, "args": None},
    }
    if slice_col is not None:
        # Add custom groupings
        custom_groups = get_groupings(df, slice_col, max_slice_bins, slice_bounds)
        groupings.update(custom_groups)
        if slice_col not in cols_to_show:
            cols_to_show.append(slice_col)

    # Fetch the results table
    df = get_concept_col_df(df, score_df, concepts, doc_id_col, doc_col, score_col, cols_to_show)
    df[doc_id_col] = df[doc_id_col].astype(str)  # Ensure doc_id_col is string type
    # cb = self.get_codebook_info()

    concept_cts = {}
    slice_cts = {}
    concept_names = [c.name for c in concepts.values()]
    df["Outlier"] = [is_outlier(row, concept_names, outlier_threshold) for _, row in df.iterrows()]
    concept_names.append("Outlier")
    concept_cts["Outlier"] = sum(df["Outlier"])

    item_metadata = {}
    matrix_df_rows = []
    item_df = None
    item_df_wide = None

    df_bullets[doc_id_col] = df_bullets[doc_id_col].astype(str)
    df_bullets = df_bullets.groupby(doc_id_col).agg(lambda x: list(x)).reset_index()

    # Rationale df
    rationale_col = "score rationale"
    highlight_col = "highlight"
    rationale_df = score_df[[doc_id_col, "concept_name", "rationale", highlight_col]]
    rationale_df.rename(columns={"rationale": rationale_col}, inplace=True)
    rationale_df[doc_id_col] = rationale_df[doc_id_col].astype(str)

    # Prep data for each group
    for group_name, group_filtering in groupings.items():
        filter_x = group_filtering["x"]
        filter_func = group_filtering["fn"]
        filter_args = group_filtering["args"]
        if filter_func is None:
            group_matches = [True] * len(df)
        else:
            group_matches = [filter_func(x, *filter_args) for x in df[filter_x].tolist()]
        cur_df = df[group_matches]
        slice_cts[group_name] = len(cur_df)

        def get_text_col_and_rename(orig_df, doc_id_col, new_col_name):
            # Internal helper to get the text column and rename it
            candidate_cols = [c for c in orig_df.columns if c != doc_id_col]
            if len(candidate_cols) != 1:
                raise ValueError(f"Expected 1 text column, got {len(candidate_cols)}")
            orig_col_name = candidate_cols[0]
            orig_df = orig_df.rename(columns={orig_col_name: new_col_name})
            return orig_df

        # Match with filtered example text
        filtered_ex_col = "quotes"
        df_filtered = get_text_col_and_rename(df_filtered, doc_id_col, new_col_name=filtered_ex_col)
        df_filtered[doc_id_col] = df_filtered[doc_id_col].astype(str)
        cur_df = cur_df.merge(df_filtered, on=doc_id_col, how="left")

        # Match with bullets
        bullets_col = "text bullets"
        df_bullets = get_text_col_and_rename(df_bullets, doc_id_col, new_col_name=bullets_col)
        cur_df = cur_df.merge(df_bullets, on=doc_id_col, how="left")

        # Item df
        item_df_cols = [doc_id_col, doc_col, filtered_ex_col, bullets_col] + cols_to_show
        cur_item_df = pd.melt(cur_df, id_vars=item_df_cols, value_vars=concept_names, var_name="concept", value_name="concept score")
        cur_item_df["id"] = group_name # cols: ex_col, concept, concept_score, id (cluster id)
        cur_item_df["concept_score_orig"] = [clean_score(x, threshold) for x in cur_item_df["concept score"].tolist()]
        cur_item_df["concept score"] = [format_scores(x) for x in cur_item_df["concept_score_orig"].tolist()]

        # Format bullets
        cur_item_df[bullets_col] = [format_bullets(orig) for orig in cur_item_df[bullets_col]]

        # Add rationale
        cur_item_df = cur_item_df.merge(rationale_df, left_on=[doc_id_col, "concept"], right_on=[doc_id_col, "concept_name"], how="left")

        # Add highlight styling
        if show_highlights:
            cur_item_df[doc_col] = cur_item_df.apply(lambda x: format_highlight(x[doc_col], x[highlight_col], x["concept_score_orig"]), axis=1)

        # Format text
        cur_df[doc_col] = [textwrap.fill(orig, width=50) for orig in cur_df[doc_col]]

        if debug:
            cur_item_df = cur_item_df[["id", "concept", "concept score", "concept_score_orig", doc_col, filtered_ex_col, bullets_col, rationale_col] + cols_to_show]
        else:
            cur_item_df = cur_item_df[["id", "concept", "concept score", "concept_score_orig", doc_col, bullets_col, rationale_col] + cols_to_show]
        if item_df is None:
            item_df = cur_item_df
        else:
            item_df = pd.concat([item_df, cur_item_df])
        
        # Item wide df
        cur_item_df_wide = cur_df.copy()
        cur_item_df_wide["id"] = group_name
        for concept in concept_names:
            cur_item_df_wide[concept] = [clean_score(x, threshold) for x in cur_item_df_wide[concept].tolist()]
            cur_item_df_wide[concept] = [format_scores(x) for x in cur_item_df_wide[concept].tolist()]

        cols_to_include = ["id", doc_col] + cols_to_show + concept_names
        cur_item_df_wide = cur_item_df_wide[cols_to_include]
        if item_df_wide is None:
            item_df_wide = cur_item_df_wide
        else:
            item_df_wide = pd.concat([item_df_wide, cur_item_df_wide])

        # Metadata
        cluster_avg_overall_score = NAN_SCORE  # TEMPLATE SCORE for datasets without scores
        item_metadata[group_name] = {
            "Slice size": f"{len(cur_df)} documents",
        }

        # Matrix df
        for concept in concept_names:
            cur_scores = [clean_score(x, threshold) for x in cur_df[concept].tolist()]
            if len(cur_scores) > 0:
                matches = [x for x in cur_scores if x > outlier_threshold]
                n_matches = len(matches)
            else:
                n_matches = 0
            
            if group_name == "All":
                concept_cts[concept] = n_matches

            matrix_row = [group_name, n_matches, group_name, cluster_avg_overall_score, concept]
            matrix_df_rows.append(matrix_row)

    matrix_df = pd.DataFrame(matrix_df_rows, columns=["id", "value", "example", "_my_score", "concept"])

    # Perform normalization
    def calc_norm_by_slice(row):
        g = row["id"]
        slice_ct = slice_cts[g]
        if slice_ct == 0:
            return 0
        return row["value"] / slice_ct
        
    def calc_norm_by_concept(row, val_col):
        c = row["concept"]
        concept_ct = concept_cts[c]
        if concept_ct == 0:
            return 0
        return row[val_col] / concept_ct

    # Add absolute count
    matrix_df["n"] = [row["value"] for _, row in matrix_df.iterrows()]
    if norm_by == "slice":
        # Normalize by slice
        matrix_df["value"] = [calc_norm_by_slice(row) for _, row in matrix_df.iterrows()]
    elif norm_by == "concept":
        # Normalize by concept
        matrix_df["value"] = [calc_norm_by_concept(row, "value") for _, row in matrix_df.iterrows()]
    
    # Replace any 0s with NAN_SCORE
    matrix_df["value"] = [NAN_SCORE if x == 0 else x for x in matrix_df["value"].tolist()]

    # Metadata
    def get_concept_metadata(c):
        ex_ids = c.example_ids
        
        if len(ex_ids) > 0:
            cur_df = score_df[(score_df["concept_id"] == c.id) & (score_df[doc_id_col].isin(ex_ids))]
            ex = cur_df[highlight_col].unique().tolist()
        else:
            ex = []
        res = {
            "Criteria": f"<br>{c.prompt}",
            "Summary": f"<br>{c.summary}",
            "Concept matches": f"{concept_cts[c.name]} documents",
            "Representative examples": f"{format_bullets(ex, add_quotes=True)}"
        }
        return res
    
    concept_metadata = {c.name: get_concept_metadata(c) for c in concepts.values()}
    concept_metadata["Outlier"] = {
        "Criteria": OUTLIER_CRITERIA,
        "Concept matches": f"{concept_cts['Outlier']} documents",
    }
    metadata_dict = {
        "items": item_metadata,
        "concepts": concept_metadata,
    }

    return matrix_df, item_df, item_df_wide, metadata_dict

# Generates the in-notebook visualization for concept induction results
#
# Input:
# - in_df: DataFrame (includes columns: doc_id_col, doc_col)
# - score_df: DataFrame (columns: doc_id_col, text, concept_id, concept_name, concept_prompt, score, highlight)
# - doc_col: string (column name for full document text)
# - doc_id_col: string (column name for document ID)
# - score_col: string (column name for concept score)
# - df_filtered: DataFrame (columns: doc_id_col, text_col)
# - df_bullets: DataFrame (columns: doc_id_col, text_col)
# - concepts: dict (concept_id -> Concept object)
#
# Parameters:
# - cols_to_show: list of strings (column names to show in visualization)
# - slice_col: string (column name to slice by)
# - max_slice_bins: int (maximum number of slices to show)
# - slice_bounds: list of numbers (manual boundaries for slices)
# - show_highlights: boolean (whether to show highlights)
# - norm_by: string (column name to normalize by; either "slice" or "concept")
# - debug: boolean (whether to print debug statements)
def visualize(in_df, score_df, doc_col, doc_id_col, score_col, df_filtered, df_bullets, concepts, cols_to_show=[], slice_col=None, max_slice_bins=None, slice_bounds=None, show_highlights=False, norm_by=None, debug=False):
    matrix_df, item_df, item_df_wide, metadata_dict = prep_vis_dfs(in_df, score_df, doc_id_col, doc_col, score_col, df_filtered, df_bullets, concepts, cols_to_show=cols_to_show, slice_col=slice_col, max_slice_bins=max_slice_bins, slice_bounds=slice_bounds,show_highlights=show_highlights, norm_by=norm_by, debug=debug)

    data = matrix_df.to_json(orient='records')
    data_items = item_df.to_json(orient='records')
    data_items_wide = item_df_wide.to_json(orient='records')
    md = json.dumps(metadata_dict)
    if slice_col is None:
        slice_col = ""
    if norm_by is None:
        norm_by = ""
    w = MatrixWidget(
        data=data, 
        data_items=data_items,
        data_items_wide=data_items_wide, 
        metadata=md,
        slice_col=slice_col,
        norm_by=norm_by,
    )
    return w, matrix_df, item_df, item_df_wide

def get_select_widget(concepts_json):
    w = ConceptSelectWidget(
        data=concepts_json,
    )
    return w

# Edits an existing concept with the specified ID
# Input: 
# - concepts: dict (concept_id -> concept dict) with existing set of concepts
# - concept_id: string (ID of the concept to edit)
# Parameters:
# - new_name: string (Optional, new name of the concept)
# - new_prompt: string (Optional, new prompt for the concept)
# - new_ex_ids: list of strings (Optional, IDs of examples that match the new concept)
# Output:
# - concepts: dict (concept_id -> concept dict) with the concept edited
def edit_concept(concepts, concept_id, new_name=None, new_prompt=None, new_ex_ids=None):
    # Update dictionary
    cur_concept = concepts[concept_id]
    if new_name is not None:
        cur_concept["name"] = new_name
    if new_prompt is not None:
        cur_concept["prompt"] = new_prompt
    if new_ex_ids is not None:
        cur_concept["example_ids"] = new_ex_ids
        # TODO: differentiate between replacing and appending example IDs
    concepts[concept_id] = cur_concept

    # TODO: handle concept_df
    return concepts
