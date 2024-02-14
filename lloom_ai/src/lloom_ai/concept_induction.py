# Main concept induction library functions
# =================================================

# Imports
import yaml
import pandas as pd
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
    from llm import multi_query_gpt_wrapper
    from prompts import *
    from __init__ import MatrixWidget
else:
    # uses current package visibility
    from .llm import multi_query_gpt_wrapper
    from .prompts import *
    from .__init__ import MatrixWidget

# CONSTANTS ================================
NAN_SCORE = -0.01  # Numerical score to use in place of NaN values for matrix viz

# SESSION class ================================

class Session:
    def __init__(
        self,
        df: pd.DataFrame,
        ex_id_col: str,
        ex_col: str,
        score_col: str = None,
        save_path: str = None,
        debug: bool = False,
    ):
        # General properties
        self.model = "gpt-3.5-turbo"
        self.use_base_api = True

        if save_path is None:
            # Automatically set using timestamp
            t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            save_path = f"./exports/{t}"
        self.save_path = save_path

        # Dataset properties
        self.df = df
        self.ex_id_col = ex_id_col
        self.ex_col = ex_col
        self.score_col = score_col

        # Stored results
        self.concepts = {}  # maps from concept_id to Concept 
        self.results = {}  # maps from result_id to Result 
        self.llm_cache = {} # Cache from hashed prompts to LLM results
        self.prompt_cache = {} # Cache from hashed prompts to full prompts
        self.cluster_cache = {} # Cache from data groups to clustering results
        self.debug = debug  # Whether to run LLM calls in debug mode (fetching from cache)

        # Qual coding flow
        self.all_args = {} # Maps from run id to log of all args in flow
        self.filtered_ex = None  # Stores Result of examples_to_filtered_examples
        self.bullets = None  # Stores Result of examples_to_bulletpoints
        self.final_summary_df = None  # Stores final summary dataframe
        self.cost = []  # Stores Result of cost_estimation
        self.tokens = {
            "in_tokens": [],
            "out_tokens": [],
        }
        self.time = {}  # Stores time required for each step
    
    def save(self):
        # Saves current session to file
        t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        cur_path = f"{self.save_path}__{t}.pkl"
        with open(cur_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved session to {cur_path}")


    # VIEW CONCEPTS AND RESULTS ================================
    def show_concepts(self):
        rows = []
        for c_id, c in self.concepts.items():
            row = [c_id, c.name, c.prompt, c.is_active]
            rows.append(row)
        out_df = pd.DataFrame(rows, columns=["id", "name", "prompt", "is_active"])
        return out_df
    
    def __get_active_concept_dict(self, check_show_in_matrix: bool = False, exclude_score_col: bool = False):
        # Return dict from concept ID to concept for concepts that are active and have df results
        def check_concept(c):
            if exclude_score_col:
                # Exclude score column
                return c.is_active and c.result is not None and c.show_in_matrix and c.name != self.score_col
            elif check_show_in_matrix:
                # Check whether concept is set to show in matrix
                return c.is_active and c.result is not None and c.show_in_matrix
            else:
                # Default: check whether concept is active and has df results
                return c.is_active and c.result is not None
            
        active_concepts = {c_id: c for c_id, c in self.concepts.items() if check_concept(c)}
        return active_concepts

    def __process_df(self, df, show_highlight=True):
        for col in [self.ex_col, "quote"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x).encode('utf-8', 'replace').decode('utf-8'))
        df = df.astype({self.ex_id_col: 'string'})
        if not show_highlight and "quote" in df.columns:
            df = df.drop(columns=["quote"])
        return df
    
    def __get_results_table(self, active_concepts=None, cols_to_add=[], use_readable_names: bool = False, check_show_in_matrix: bool = False, show_highlight: bool = True, only_show_sample=False):
        # Returns a merged dataframe of all concept results
        # Can be filtered with check_show_in_matrix to only display concepts that have show_in_matrix set to True (currently excludes residual summaries and other riffing prompts)
        if only_show_sample:
            ex_ids = [str(cur_id) for cur_id in self.bullets.df[self.ex_id_col].tolist()]
            df = self.df.copy()
            df[self.ex_id_col] = df[self.ex_id_col].astype(str)
            df = df[df[self.ex_id_col].isin(ex_ids)].copy()
        else:
            df = self.df.copy()

        if active_concepts is None:
            active_concepts = self.__get_active_concept_dict(check_show_in_matrix)
        active_concept_dfs = {c.id: self.results[c.result].df for c in active_concepts.values()}

        if use_readable_names:
            cols_to_show = [self.ex_id_col, self.ex_col]
            if len(cols_to_add) > 0:
                cols_to_show.extend(cols_to_add)
            cur_df = df[cols_to_show].copy()
        else:
            cur_df = df.copy()

        cur_df = self.__process_df(cur_df)
        for c_id, res_df in active_concept_dfs.items():
            res_df = self.__process_df(res_df, show_highlight)
            # Add concept results to df
            if use_readable_names:
                res_df = res_df.rename(columns={
                    c_id: active_concepts[c_id].name, 
                    "rationale": f"{active_concepts[c_id].name}__rationale", 
                    "quote": f"{active_concepts[c_id].name}__quote"
                })
            cur_df = cur_df.merge(res_df, on=self.ex_id_col, how="left")
        return cur_df, active_concepts

    def show_results(self, active_concepts=None, cols_to_add=[], show_highlight=True, only_show_sample=False):
        cur_df, _ = self.__get_results_table(active_concepts=active_concepts, cols_to_add=cols_to_add, use_readable_names=True, show_highlight=show_highlight, only_show_sample=only_show_sample)
        return cur_df


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

def cluster_helper(sess, in_df, text_col, min_cluster_size, cur_col_name):
    if sess.debug:
        # Fetch cached cluster results
        return sess.cluster_cache[cur_col_name]

    # OpenAI embeddings with HDBSCAN clustering
    embedding_model = OpenAIBackend("text-embedding-ada-002")

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

    id_vals = in_df[sess.ex_id_col].tolist()
    text_vals = in_df[text_col].tolist()
    clusters, probs = topic_model.fit_transform(text_vals)
    rows = list(zip(id_vals, text_vals, clusters)) # id_col, text_col, cluster_id_col
    cluster_df = pd.DataFrame(rows, columns=[sess.ex_id_col, text_col, cur_col_name])
    cluster_df = cluster_df.sort_values(by=[cur_col_name])
    
    # Store in cache
    sess.cluster_cache[cur_col_name] = cluster_df
    return cluster_df

def save_timing(sess, start, cur_col_name=None):
    elapsed = time.time() - start
    print(f"Total time: {elapsed:0.2f} sec")
    if cur_col_name is not None:
        sess.time[cur_col_name] = elapsed

def cost_estimation(sess, results, model_name):
    # https://openai.com/pricing
    # https://platform.openai.com/docs/models/gpt-3-5
    COSTS = {
        "gpt-3.5-turbo": [0.0010/1000, 0.002/1000],
        "gpt-3.5-turbo-1106": [0.0010/1000, 0.002/1000],
        "gpt-4": [0.03/1000, 0.06/1000]
    }
    # Cost estimation
    in_tokens = [res.llm_output["token_usage"]["prompt_tokens"] for res in results]
    out_tokens = [res.llm_output["token_usage"]["completion_tokens"] for res in results]
    in_token_sum = np.sum(in_tokens)
    out_token_sum = np.sum(out_tokens)
    in_token_cost = in_token_sum * COSTS[model_name][0]
    out_token_cost = out_token_sum * COSTS[model_name][1]
    total_cost = in_token_cost + out_token_cost
    print(f"Total: {total_cost} | In: {in_token_cost} | Out: {out_token_cost}")
    sess.cost.append(total_cost)
    sess.tokens["in_tokens"].append(in_token_sum)
    sess.tokens["out_tokens"].append(out_token_sum)

def get_orig_text_df(sess):
    text_df = sess.df.copy()
    return text_df[[sess.ex_id_col, sess.ex_col]]

def filter_empty_rows(df, text_col_name):
    # Remove rows where the specified column is empty
    df_out = df[df[text_col_name].apply(lambda x: len(x) > 0)]
    return df_out

# CORE functions ================================

# Input: 
# - sess: Session object
# - text_df: DataFrame (columns: doc_id, text)
# Parameters: n_quotes, seed
# Output: quote_df (columns: doc_id, quote)
async def distill_filter(sess, i, text_df, n_quotes=3, seed=None):
    # Filtering operates on provided text
    start = time.time()
    cur_col_name = f"{i}_distill_filter"
    if text_df is None:
        text_df = get_orig_text_df(sess)

    # Filter to non-empty rows
    text_df = filter_empty_rows(text_df, sess.ex_col)

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
        } for ex in text_df[sess.ex_col].tolist()
    ]
    
    # Run prompts
    prompt_template = filter_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, sess.model, sess)

    # Process results
    ex_ids = [ex_id for ex_id in text_df[sess.ex_id_col].tolist()]
    for ex_id, res in zip(ex_ids, res_text):
        cur_filtered_list = json_load(res, top_level_key="relevant_quotes")
        if cur_filtered_list is not None:
            cur_filtered = "\n".join(cur_filtered_list)
            filtered_ex.append(cur_filtered)
            rows.append([ex_id, cur_filtered])
    quote_df = pd.DataFrame(rows, columns=[sess.ex_id_col, cur_col_name])
    cost_estimation(sess, res_full, sess.model)

    save_timing(sess, start, cur_col_name)
    return quote_df


# Input: text_df (columns: doc_id, text) 
#   --> text could be original or filtered (quotes)
# Parameters: n_bullets, n_words_per_bullet, seed
# Output: bullet_df (columns: doc_id, bullet)
async def distill_summarize(sess, i, text_df, text_col, n_bullets="2-4", n_words_per_bullet="5-8", seed=None):
    # Summarization operates on text_col
    start = time.time()
    cur_col_name = f"{i}_distill_summarize"
    if text_df is None:
        text_df = get_orig_text_df(sess)
        text_col = sess.ex_col
    
    # Filter to non-empty rows
    text_df = filter_empty_rows(text_df, text_col)

    # Prepare prompts
    rows = []
    arg_dicts = []
    all_ex_ids = []

    if seed is not None:
        seeding_phrase = f"related to {seed}"
    else:
        seeding_phrase = ""
    for _, row in text_df.iterrows():
        ex = row[text_col]
        ex_id = row[sess.ex_id_col]
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
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, sess.model, sess)

    # Process results
    for ex_id, res in zip(all_ex_ids, res_text):
        cur_bullets_list = json_load(res, top_level_key="bullets")
        if cur_bullets_list is not None:
            for bullet in cur_bullets_list:
                # Expand each bullet into its own row
                rows.append([ex_id, bullet])
    bullet_df = pd.DataFrame(rows, columns=[sess.ex_id_col, cur_col_name])
    cost_estimation(sess, res_full, sess.model)

    save_timing(sess, start, cur_col_name)
    return bullet_df


# Input: text_df (columns: doc_id, text) 
#   --> text could be original, filtered (quotes), and/or summarized (bullets)
# Parameters: n_clusters
# Output: cluster_df (columns: doc_id, text, cluster_id)
async def cluster(sess, i, text_df, text_col, min_cluster_size=None, batch_size=20, randomize=False):
    # Clustering operates on text_col
    start = time.time()
    cur_col_name = f"{i}_cluster"

    # Filter to non-empty rows
    text_df = filter_empty_rows(text_df, text_col)

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
        cluster_df[cur_col_name] = cluster_ids
    else:
        # Cluster and group by clusters
        cluster_df = cluster_helper(sess, text_df, text_col, min_cluster_size=min_cluster_size, cur_col_name=cur_col_name)

    save_timing(sess, start, cur_col_name)
    return cluster_df


def dict_to_json(examples):
    # Internal helper to convert examples to json for prompt
    examples_json = json.dumps(examples)
    # Escape curly braces to avoid the system interpreting as template formatting
    examples_json = examples_json.replace("{", "{{")
    examples_json = examples_json.replace("}", "}}")
    return examples_json

# Input: cluster_df (columns: doc_id, text, cluster_id)
# Parameters: n_concepts
# Output: 
# - concepts: dict (concept_id -> concept dict)
# - concept_df: DataFrame (columns: doc_id, text, concept_id, concept_name, concept_prompt)
async def synthesize(sess, i, cluster_df, text_col, cluster_id_col, n_concepts=None, batch_size=None, model_name=None, verbose=True, pattern_phrase="unifying pattern", seed=None):
    # Synthesis operates on "text" column for each cluster_id
    # Concept object is created for each concept
    start = time.time()
    cur_col_name = f"{i}_synthesize"
    if model_name is None:
        model_name = sess.model
    
    # Filter to non-empty rows
    cluster_df = filter_empty_rows(cluster_df, text_col)

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
    ex_id_to_ex = {(str(row[sess.ex_id_col]), row[cluster_id_col]): row[text_col] for _, row in cluster_df.iterrows()}  # Map example IDs to example text
    for cluster_id in cluster_ids:
        # Iterate over cluster IDs to get example sets
        cur_df = cluster_df[cluster_df[cluster_id_col] == cluster_id]
        cluster_dfs[cluster_id] = cur_df
        if batch_size is not None:
            # Split into batches
            n_batches = math.ceil(len(cur_df) / batch_size)
            for i in range(n_batches):
                cur_batch_df = cur_df.iloc[i*batch_size:(i+1)*batch_size]
                ex_dicts = [{"example_id": row[sess.ex_id_col], "example": row[text_col]} for _, row in cur_batch_df.iterrows()]
                ex_dicts_json = dict_to_json(ex_dicts)
                arg_dict = {
                    "examples": ex_dicts_json,
                    "n_concepts_phrase": get_n_concepts_phrase(cur_df),
                    "seed_phrase": seed_phrase
                }
                arg_dicts.append(arg_dict)
        else:
            # Handle unbatched case
            ex_dicts = [{"example_id": row[sess.ex_id_col], "example": row[text_col]} for _, row in cur_df.iterrows()]
            ex_dicts_json = dict_to_json(ex_dicts)
            arg_dict = {
                "examples": ex_dicts_json,
                "n_concepts_phrase": get_n_concepts_phrase(cur_df),
                "seed_phrase": seed_phrase
            }
            arg_dicts.append(arg_dict)

    # Run prompts
    prompt_template = synthesize_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name, sess)

    # Process results
    concepts = {}
    rows = []
    for cur_cluster_id, res in zip(cluster_ids, res_text):
        cur_concepts = json_load(res, top_level_key="patterns")
        if cur_concepts is not None:
            # concepts.extend(cur_concepts)
            # cur_examples_dict = {ex_to_id[cur_ex]: cur_ex for cur_ex in cur_examples}
            
            for concept in cur_concepts:
                concept_id = str(uuid.uuid4())
                concepts[concept_id] = concept
                ex_ids = concept["example_ids"]
                ex_ids = set(ex_ids) # remove duplicates
                for ex_id in ex_ids:
                    # doc_id, text, concept_id, concept_name, concept_prompt
                    cur_key = (ex_id, cur_cluster_id)
                    if cur_key in ex_id_to_ex:
                        row = [ex_id, ex_id_to_ex[cur_key], concept_id, concept["name"], concept["prompt"]]
                        rows.append(row)
            if verbose:
                examples = cluster_dfs[cur_cluster_id][text_col].tolist()
                concepts_formatted = pretty_print_dict_list(cur_concepts)
                print(f"\n\nInput examples: {examples}\nOutput concepts: {concepts_formatted}")
    # doc_id, text, concept_id, concept_name, concept_prompt
    concept_df = pd.DataFrame(rows, columns=[sess.ex_id_col, text_col, cur_col_name, f"{cur_col_name}_name", f"{cur_col_name}_prompt"])
    cost_estimation(sess, res_full, model_name)

    save_timing(sess, start, cur_col_name)
    return concepts, concept_df

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
async def review(sess, i, concepts, concept_df, concept_col, concept_col_prefix, n_concepts=None, model_name=None, debug=True):
    # Model is asked to review the provided set of concepts
    if model_name is None:
        model_name = sess.model

    concepts_out, concept_df_out, removed = await review_remove(sess, i, concepts, concept_df, concept_col, concept_col_prefix, model_name=model_name)
    concepts_out, concept_df_out, merged = await review_merge(sess, i, concepts_out, concept_df_out, concept_col, concept_col_prefix, model_name=model_name)

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
async def review_remove(sess, i, concepts, concept_df, concept_col, concept_col_prefix, model_name=None):
    concepts = concepts.copy()  # Make a copy of the concepts dict to avoid modifying the original
    start = time.time()
    cur_step_name = f"{i}_refine_remove"
    concept_name_col = f"{concept_col_prefix}_name"
    if model_name is None:
        model_name = sess.model

    concepts_list = concept_df[concept_col].tolist()
    concepts_list = [f"- {c}" for c in concepts_list]
    concepts_list_str = "\n".join(concepts_list)
    arg_dicts = [{
        "themes": concepts_list_str,
    }]

    # Run prompts
    prompt_template = review_remove_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name, sess)

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

    cost_estimation(sess, res_full, model_name)
    save_timing(sess, start, cur_step_name)
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
async def review_merge(sess, i, concepts, concept_df, concept_col, concept_col_prefix, model_name=None):
    concepts = concepts.copy()  # Make a copy of the concepts dict to avoid modifying the original
    start = time.time()
    cur_step_name = f"{i}_refine_merge"
    concept_name_col = f"{concept_col_prefix}_name"
    concept_prompt_col = f"{concept_col_prefix}_prompt"
    if model_name is None:
        model_name = sess.model

    concepts_list = concept_df[concept_col].tolist()
    concepts_list = [f"- {c}" for c in concepts_list]
    concepts_list_str = "\n".join(concepts_list)
    arg_dicts = [{
        "themes": concepts_list_str,
    }]

    # Run prompts
    prompt_template = review_merge_prompt
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name, sess)

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

    cost_estimation(sess, res_full, model_name)
    save_timing(sess, start, cur_step_name)
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
    OUT_COLS = ["doc_id", "text", "concept_id", "concept_name", "concept_prompt", "score", "rationale", "highlight"]
    concept_name = concept["name"]
    concept_prompt = concept["prompt"]
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
        
        out_df = pd.DataFrame(rows, columns=OUT_COLS)
        return out_df
    else:
        out_df = in_df.copy()
        out_df["doc_id"] = out_df[doc_id_col]
        out_df["text"] = out_df[text_col]
        out_df["concept_id"] = concept_id
        out_df["concept_name"] = concept_name
        out_df["concept_prompt"] = concept_prompt
        out_df["score"] = NAN_SCORE
        out_df["rationale"] = ""
        out_df["highlight"] = ""
        return out_df[OUT_COLS]

# Performs scoring for one concept
async def score_helper(sess, concept, batch_i, concept_id, df, text_col, doc_id_col, model_name, batch_size, get_highlights):
    # TODO: add support for only a sample of examples
    # TODO: set consistent concept IDs for reference
    # TODO: add support for highlighting

    concept_name = concept["name"]
    concept_prompt = concept["prompt"]
    concept_example_ids = concept["example_ids"]

    # Prepare batches of input arguments
    indices = range(0, len(df), batch_size)
    ex_ids = [str(x) for x in df[doc_id_col].tolist()]
    ex_id_sets = [ex_ids[i:i+batch_size] for i in indices]
    in_dfs = [df[df[doc_id_col].isin(cur_ex_ids)] for cur_ex_ids in ex_id_sets]
    arg_dicts = [
        get_ex_batch_args(df, text_col, doc_id_col, concept_name, concept_prompt) for df in in_dfs
    ]

    # Run prompts in parallel to score each example
    if get_highlights:
        prompt_template = score_highlight_prompt
    else:
        prompt_template = score_no_highlight_prompt
    results, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name, sess, batch_num=batch_i)

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

    cost_estimation(sess, res_full, model_name)
    return score_df

# Performs scoring for all concepts
# Input: concepts, text_df (columns: doc_id, text)
#   --> text could be original, filtered (quotes), and/or summarized (bullets)
# Parameters: threshold
# Output: score_df (columns: doc_id, text, concept_id, concept_name, concept_prompt, score, highlight)
async def score(sess, i, text_df, text_col, doc_id_col, concepts, model_name="gpt-3.5-turbo", batch_size=5, get_highlights=False):
    # Scoring operates on "text" column for each concept
    # Concept is added to session after scoring is complete
    start = time.time()
    cur_step_name = f"{i}_score"

    text_df = text_df.copy()
    # Filter to non-empty rows
    text_df = filter_empty_rows(text_df, text_col)

    text_df[doc_id_col] = text_df[doc_id_col].astype(str)
    tasks = [score_helper(sess, concept, concept_i, concept_id, text_df, text_col, doc_id_col, model_name, batch_size, get_highlights) for concept_i, (concept_id, concept) in enumerate(concepts.items())]
    score_dfs = await tqdm_asyncio.gather(*tasks, file=sys.stdout)

    # Combine score_dfs
    score_df = pd.concat(score_dfs)

    save_timing(sess, start, cur_step_name)
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

# Input: score_df (columns: doc_id, text, concept_id, score)
# Output: text_df (columns: doc_id, text) --> filtered set of documents to run further iterations of concept induction
# Returns None if (1) there are zero remaining documents after filtering or (2) there are no changes in the number of documents after filtering
def loop(sess, score_df, text_col, debug=False):
    # Check for not-covered and covered-by-generic documents
    # Not-covered: examples that don't match any concepts
    # Covered-by-generic: examples that only match generic concepts (those which cover the majority of examples)
    # TODO: Allow users to decide on custom filtering conditions
    # TODO: Save generic concepts to session (to avoid later)
    n_initial = len(score_df[sess.ex_id_col].unique())

    underrep_ids = get_not_covered(score_df, sess.ex_id_col)
    generic_ids = get_covered_by_generic(score_df, sess.ex_id_col)
    ids_to_include = underrep_ids + generic_ids
    ids_to_include = set(ids_to_include)
    if debug:
        print(f"ids_to_include ({len(ids_to_include)}): {ids_to_include}")

    text_df = score_df.copy()
    text_df = text_df[text_df[sess.ex_id_col].isin(ids_to_include)][[sess.ex_id_col, text_col]].drop_duplicates().reset_index()

    # Stopping condition: if text_df is empty or has the same number of docs as the original df
    n_final = len(text_df[sess.ex_id_col].unique())
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
async def auto_eval(sess, items, concepts, model_name="gpt-3.5-turbo", debug=False):
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
    res_text, res_full = await multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name, sess)
    
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

    cost_estimation(sess, res_full, model_name)
    save_timing(sess, start)

    return concepts_found, concept_coverage

# Adds color-background styling for score columns to match score value
def format_scores(x: float):
    color_str = f"rgba(130, 193, 251, {x*0.5})"
    start_tag = f"<div class='score-col' style='background-color: {color_str}'>"
    return f"{start_tag}{x}</div>"

# Converts a list of bullet strings to a formatted unordered list
def format_bullets(orig):
    if (not isinstance(orig, list)) or len(orig) == 0:
        return ""
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
        concept_name = c["name"]
        c_df = score_df[score_df["concept_id"] == c_id][concept_cols_to_show]
        c_df = c_df.rename(columns={score_col: concept_name}) # Store score under concept name
        # Rename columns and remove unused columns
        cur_df[doc_id_col] = cur_df[doc_id_col].astype(str)
        c_df[doc_id_col] = c_df[doc_id_col].astype(str)
        cur_df = cur_df.merge(c_df, on=doc_id_col, how="left")
    return cur_df

# Helper function for `visualize()` to generate the underlying dataframes
# Parameters:
# - threshold: float (minimum score of positive class)
def prep_vis_dfs(df, score_df, doc_id_col, doc_col, score_col, df_filtered, df_bullets, concepts, cols_to_show, custom_groups, show_highlights, norm_by, debug=False, threshold=None, outlier_threshold=0.75):
    # TODO: codebook info

    # Handle groupings
    # Add the "All" grouping by default
    groupings = {
        "All": {"x": None, "fn": None},
    }
    if len(custom_groups) > 0:
        # Add custom groupings
        groupings.update(custom_groups)
        for group_name, group_info in custom_groups.items():
            grouping_col = group_info["x"]
            if grouping_col not in cols_to_show:
                cols_to_show.append(grouping_col)

    # Fetch the results table
    df = get_concept_col_df(df, score_df, concepts, doc_id_col, doc_col, score_col, cols_to_show)
    df[doc_id_col] = df[doc_id_col].astype(str)  # Ensure doc_id_col is string type
    # cb = self.get_codebook_info()

    concept_cts = {}
    group_cts = {}
    concept_names = [c["name"] for c in concepts.values()]
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
    rationale_col = "rationale"
    highlight_col = "highlight"
    rationale_df = score_df[[doc_id_col, "concept_name", rationale_col, highlight_col]]
    rationale_df[doc_id_col] = rationale_df[doc_id_col].astype(str)

    # Prep data for each group
    for group_name, group_filtering in groupings.items():
        filter_x = group_filtering["x"]
        filter_func = group_filtering["fn"]
        if filter_func is None:
            group_matches = [True] * len(df)
        else:
            group_matches = [filter_func(x) for x in df[filter_x].tolist()]
        cur_df = df[group_matches]
        group_cts[group_name] = len(cur_df)

        def get_text_col_and_rename(df, orig_df, doc_id_col, new_col_name):
            # Internal helper to get the text column and rename it
            candidate_cols = [c for c in orig_df.columns if c != doc_id_col]
            if len(candidate_cols) != 1:
                raise ValueError(f"Expected 1 text column, got {len(candidate_cols)}")
            orig_col_name = candidate_cols[0]
            df = df.rename(columns={orig_col_name: new_col_name})
            return df

        # Match with filtered example text
        df_filtered[doc_id_col] = df_filtered[doc_id_col].astype(str)
        cur_df = cur_df.merge(df_filtered, on=doc_id_col, how="left")
        filtered_ex_col = "quotes"
        cur_df = get_text_col_and_rename(cur_df, df_filtered, doc_id_col, new_col_name=filtered_ex_col)

        # Match with bullets
        cur_df = cur_df.merge(df_bullets, on=doc_id_col, how="left")
        bullets_col = "bullets"
        cur_df = get_text_col_and_rename(cur_df, df_bullets, doc_id_col, new_col_name=bullets_col)

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
            "Slice size": f"{len(cur_df)} examples",
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
    def calc_norm_by_group(row):
        g = row["id"]
        group_ct = group_cts[g]
        if group_ct == 0:
            return 0
        return row["value"] / group_ct
        
    def calc_norm_by_concept(row, val_col):
        c = row["concept"]
        concept_ct = concept_cts[c]
        if concept_ct == 0:
            return 0
        return row[val_col] / concept_ct

    if norm_by == "group":
        # Normalize by group
        matrix_df["value"] = [calc_norm_by_group(row) for _, row in matrix_df.iterrows()]
    else:
        # Normalize by concept
        matrix_df["value"] = [calc_norm_by_concept(row, "value") for _, row in matrix_df.iterrows()]
    
    # Replace any 0s with NAN_SCORE
    matrix_df["value"] = [NAN_SCORE if x == 0 else x for x in matrix_df["value"].tolist()]

    # Metadata
    # def format_codebook_entry(x):
    #     entry = cb[x]
    #     lo_themes = entry["lo_themes"]
    #     full_str = ""
    #     for lo_theme_name, lo_theme in lo_themes.items():
    #         cur_str = f"<li><b>{lo_theme['name']}</b>: {lo_theme['prompt']}</li>"
    #         ex_str = ""
    #         for ex in lo_theme["examples"]:
    #             ex_str += f"<li><i>\"{ex}\"</i></li>"
    #         if len(ex_str) > 0:
    #             cur_str += f"<ul>{ex_str}</ul>"
    #         full_str += f"{cur_str}<br>"
    #     full_str = f"<ul>{full_str}<ul>"
    #     return full_str
    def get_concept_metadata(c):
        # Fetch other codebook info
        # codebook_info = format_codebook_entry(c["name"]) if c["name"] in cb else None
        res = {
            "Criteria": f"<br>{c['prompt']}",
            "Concept matches": f"{concept_cts[c['name']]} examples",
        }
        # if codebook_info is not None:
        #     res["Subconcepts and examples"] = codebook_info
        return res
    
    concept_metadata = {c["name"]: get_concept_metadata(c) for c in concepts.values()}
    concept_metadata["Outlier"] = {
        "Criteria": "Did the example not match any of the above concepts?",
        "Concept matches": f"{concept_cts['Outlier']} examples",
    }
    metadata_dict = {
        "items": item_metadata,
        "concepts": concept_metadata,
    }

    return matrix_df, item_df, item_df_wide, metadata_dict

# Generates the in-notebook visualization for concept induction results
#
# Input:
# - df: DataFrame (includes columns: doc_id_col, doc_col)
# - score_df: DataFrame (columns: doc_id_col, text, concept_id, concept_name, concept_prompt, score, highlight)
# - doc_id_col: string (column name for document ID)
# - doc_col: string (column name for full document text)
# - score_col: string (column name for concept score)
# - df_filtered: DataFrame (columns: doc_id_col, text_col)
# - df_bullets: DataFrame (columns: doc_id_col, text_col)
#
# Parameters:
# - cols_to_show: list of strings (column names to show in visualization)
# - custom_groups: dict (group_name -> group information)
# - show_highlights: boolean (whether to show highlights)
# - norm_by: string (column name to normalize by; either "group" or "concept"
# - debug: boolean (whether to print debug statements)
def visualize(df, score_df, res_dict, concepts, cols_to_show=[], custom_groups={}, show_highlights=False, norm_by="concept", debug=False):
    doc_id_col=res_dict['doc_id_col']
    doc_col=res_dict['doc_col']
    score_col=res_dict['score_col']
    df_filtered=res_dict['df_filtered']
    df_bullets=res_dict['df_bullets']
    matrix_df, item_df, item_df_wide, metadata_dict = prep_vis_dfs(df, score_df, doc_id_col, doc_col, score_col, df_filtered, df_bullets, concepts, cols_to_show=cols_to_show, custom_groups=custom_groups, show_highlights=show_highlights, norm_by=norm_by, debug=debug)

    data = matrix_df.to_json(orient='records')
    data_items = item_df.to_json(orient='records')
    data_items_wide = item_df_wide.to_json(orient='records')
    md = json.dumps(metadata_dict)
    w = MatrixWidget(
        data=data, 
        data_items=data_items,
        data_items_wide=data_items_wide, 
        metadata=md
    )
    return w

# Adds a new concept with the specified name and prompt
# Input: 
# - concepts: dict (concept_id -> concept dict) with existing set of concepts
# Parameters:
# - name: string (name of the new concept)
# - prompt: string (prompt for the new concept)
# - ex_ids: list of strings (Optional, IDs of examples that match the new concept)
# Output:
# - concepts: dict (concept_id -> concept dict) with the new concept added
def add_concept(concepts, name, prompt, ex_ids=[]):
    # Update dictionary
    concept_id = str(uuid.uuid4())
    concepts[concept_id] = {
        "name": name,
        "prompt": prompt,
        "example_ids": ex_ids,
    }

    # TODO: handle concept_df
    return concepts, concept_id

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


# WRAPPER FUNCTIONS
async def concept_gen(cur_df=None, seed=None, doc_col="text", doc_id_col="doc_id", save_id="", model_name="gpt-4", args=None, debug=True):
    if args is None:
        args = {
            "filter_n_quotes": 2,
            "summ_n_bullets": "2-4",
            "cluster_batch_size1": 20,
            "synth_n_concepts1": 10,
            "cluster_batch_size2": 15,
            "synth_n_concepts2": 10,
        }

    cur_df = cur_df.sample(frac=1).reset_index()  # shuffle order
    
    # Create session
    s = Session(
        df=cur_df,
        ex_id_col=doc_id_col,  # TODO: edit for dataset (column containing example IDs)
        ex_col=doc_col,  # TODO: edit for dataset (column containing example text)
        save_path = f'../nb/exports/{save_id}',  # TODO: edit to the desired directory
        debug=False, # Set True to only use cache for LLM results
    )

    # Run concept generation
    df_filtered = await distill_filter(
        s, i=0, 
        text_df=None, 
        n_quotes=args["filter_n_quotes"],
        seed=seed,
    )
    if debug:
        print("df_filtered")
        display(df_filtered)
    
    df_bullets = await distill_summarize(
        s, i=0, 
        text_df=df_filtered, 
        text_col="0_distill_filter",
        n_bullets=args["summ_n_bullets"],
        seed=seed,
    )
    if debug:
        print("df_bullets")
        display(df_bullets)

    df_cluster = await cluster(
        s, i=0, 
        text_df=df_bullets, 
        text_col="0_distill_summarize",
        batch_size=args["cluster_batch_size1"],
    )
    if debug:
        print("df_cluster")
        display(df_cluster)
    
    concepts, df_concepts = await synthesize(
        s, i=0, 
        cluster_df=df_cluster, 
        text_col="0_distill_summarize", 
        cluster_id_col="0_cluster",
        model_name=model_name, # use specified model for this step
        n_concepts=args["synth_n_concepts1"],
        pattern_phrase="unique topic",
        seed=seed,
    )
    df_concepts["0_synthesize_namePrompt"] = df_concepts["0_synthesize_name"] + ": " + df_concepts["0_synthesize_prompt"]
    df_concepts = dedupe_concepts(df_concepts, concept_col="0_synthesize_namePrompt")

    if debug:
        # Print results
        print("=======\n\nAfter self-review")
        for k, concept_dict in concepts.items():
            concept = concept_dict["name"]
            prompt = concept_dict["prompt"]
            print(f'- Concept {k}: {concept}\n\t- Prompt: {prompt}')

    res_dict = {
        "df_filtered": df_filtered,
        "df_bullets": df_bullets, 
        "df_cluster": df_cluster,
        "concepts": concepts, 
        "df_concepts": df_concepts,
        "text_df": "df_filtered",
        "text_col": "0_distill_filter",
        "doc_col": doc_col,
        "doc_id_col": doc_id_col,
        "score_col": "score",
    }

    if debug:
        # Print final results
        print(f"=======\n\nFinal concept names ({len(concepts)}):")
        concept_dicts = [v for _, v in concepts.items()]
        for c in concept_dicts:
            print(f'- {c["name"]}')

    return s, res_dict, concepts


async def concept_score(s, res_dict, concepts, max_concepts_to_score=5, get_highlights=False):
    # Limit to max_concepts_to_score; only score those concepts
    concepts_lim = {}
    i = 0
    for k, v in concepts.items():
        if i >= max_concepts_to_score:
            break
        concepts_lim[k] = v
        i += 1
    
    # Run usual scoring
    text_df_key = res_dict["text_df"]
    score_df = await score(
        s, i=1, 
        text_df=res_dict[text_df_key], 
        text_col=res_dict["text_col"], 
        doc_id_col=res_dict["doc_id_col"],
        concepts=concepts_lim,
        get_highlights=get_highlights,
    )

    return score_df, concepts_lim
    
async def concept_add(s, res_dict, concepts, score_df, name, prompt, get_highlights=True):
    concepts, concept_id = add_concept(concepts, name, prompt)

    # Run scoring
    cur_concept_dict = {concept_id: concepts[concept_id]}
    cur_score_df, _ = await concept_score(s, res_dict, cur_concept_dict, max_concepts_to_score=1, get_highlights=get_highlights)

    # Add results to score df
    score_df = pd.concat([score_df, cur_score_df]).reset_index()
    return concepts, score_df

# CONCEPT SELECTION WIDGET

def multi_checkbox_widget(options_dict):
    """ Widget with a search field and lots of checkboxes """
    search_widget = widgets.Text()
    output_widget = widgets.Output()
    options = [x for x in options_dict.values()]
    options_layout = widgets.Layout(
        overflow='auto',
        border='1px solid grey',
        width='300px',
        height='300px',
        flex_flow='column',
        display='flex'
    )
    
    options_widget = widgets.VBox(options, layout=options_layout)
    multi_select = widgets.VBox([search_widget, options_widget])

    @output_widget.capture()
    def on_checkbox_change(change):
        options_widget.children = sorted([x for x in options_widget.children], key = lambda x: x.value, reverse = True)
        
    for checkbox in options:
        checkbox.observe(on_checkbox_change, names="value")

    # Wire the search field to the checkboxes
    @output_widget.capture()
    def on_text_change(change):
        search_input = change['new']
        if search_input == '':
            # Reset search field
            new_options = sorted(options, key = lambda x: x.value, reverse = True)
        else:
            # Filter by search field using difflib.
            close_matches = [x for x in list(options_dict.keys()) if str.lower(search_input.strip('')) in str.lower(x)]
            new_options = sorted(
                [x for x in options if x.description in close_matches], 
                key = lambda x: x.value, reverse = True
            )
        options_widget.children = new_options

    search_widget.observe(on_text_change, names='value')
    display(output_widget)
    return multi_select

def concept_select(concepts):
    options_dict = {
        c['name']: widgets.Checkbox(
            # description=f"{c['name']}: {c['prompt']}", 
            description=c['name'],
            value=False,
            style={"description_width":"0px"}
        ) for c_id, c in concepts.items()
    }
    ui = multi_checkbox_widget(options_dict)
    return ui

def get_selected(w, concepts):
    c_names = [c.description for c in w.children[1].children if c.value]
    selected_concepts = {c_id: c for c_id, c in concepts.items() if c['name'] in c_names}
    return selected_concepts
