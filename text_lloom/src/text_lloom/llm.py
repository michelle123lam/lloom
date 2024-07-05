"""
llm.py
------
This file contains utility functions for processing calls to LLMs.
"""

# IMPORTS ================================
import numpy as np
import asyncio

# MODEL CLASSES ================================
class Model:
    # Specification to run LLooM operator with a specific large language model
    # - name: str, name of the model (ex: "gpt-3.5-turbo")
    # - setup_fn: function, function to set up the LLM client
    # - fn: function, function to call the model (i.e., to run LLM prompt)
    # - cost: float, cost per token
    # - rate_limit: tuple, (n_requests, wait_time_secs)
    # - context_window: int, total tokens shared between input and output
    # - api_key: str, API key
    def __init__(self, name, setup_fn, fn, cost, rate_limit, context_window, api_key, **args):
        self.name = name
        self.setup_fn = setup_fn
        self.fn = fn
        self.cost = cost
        self.rate_limit = rate_limit
        self.context_window = context_window
        self.client = setup_fn(api_key)
        self.args = args


class EmbedModel:
    # Specification to run LLooM operator with a specific embedding model
    # - name: str, name of the model (ex: "text-embedding-ada-002")
    # - setup_fn: function, function to set up the embedding client
    # - fn: function, function to call the model (i.e., to fetch embedding)
    # - cost: float, cost per token
    # - batch_size: int, maximum batch size for embeddings (None for no batching)
    # - api_key: str, API key
    def __init__(self, name, setup_fn, fn, cost, batch_size, api_key, **args):
        self.name = name
        self.setup_fn = setup_fn
        self.fn = fn
        self.cost = cost
        self.batch_size = batch_size
        self.client = setup_fn(api_key)
        self.args = args


# def get_token_estimate(text, model_name):
#     # Fetch the number of tokens used by a prompt
#     encoding = tiktoken.encoding_for_model(model_name)
#     tokens = encoding.encode(text)
#     num_tokens = len(tokens)
#     return num_tokens

# def calc_cost_by_tokens(model_name, in_tokens, out_tokens):
#     # Calculate cost with the tokens and model name
#     in_cost = in_tokens * COSTS[model_name][0]
#     out_cost = out_tokens * COSTS[model_name][1]
#     return in_cost, out_cost


# CUSTOM LLM API WRAPPERS ================================

# Wrapper for calling the base OpenAI API
async def base_api_wrapper(cur_prompt, model):
    res = await model.fn(model, cur_prompt)
    return res

# Internal function making calls to LLM; runs a single LLM query
async def multi_query_gpt(model, prompt_template, arg_dict, batch_num=None, wait_time=None, debug=False):
    if wait_time is not None:
        if debug:
            print(f"Batch {batch_num}, wait time {wait_time}")
        await asyncio.sleep(wait_time)  # wait asynchronously

    try: 
        prompt = prompt_template.format(**arg_dict)
        res = await base_api_wrapper(prompt, model)
    except Exception as e:
        print("Error", e)
        return None
    
    return res

# Run multiple LLM queries
async def multi_query_gpt_wrapper(prompt_template, arg_dicts, model, batch_num=None, batched=True, debug=False):
    if debug:
        print("model_name", model.name)
    
    rate_limit = model.rate_limit
    if not batched:
        # Non-batched version
        tasks = [multi_query_gpt(model, prompt_template, args) for args in arg_dicts]
    else:
        # Batched version
        n_requests, wait_time_secs = rate_limit
        tasks = []
        arg_dict_batches = [arg_dicts[i:i + n_requests] for i in range(0, len(arg_dicts), n_requests)]
        for inner_batch_num, cur_arg_dicts in enumerate(arg_dict_batches):
            if batch_num is None:
                wait_time = wait_time_secs * inner_batch_num
            else:
                wait_time = wait_time_secs * batch_num
            if debug:
                wait_time = 0 # Debug mode
            cur_tasks = [multi_query_gpt(model, prompt_template, arg_dict=args, batch_num=batch_num, wait_time=wait_time) for args in cur_arg_dicts]
            tasks.extend(cur_tasks)

    res_full = await asyncio.gather(*tasks)

    # TODO: edit so res_text is only text and tokens has token counts
    res_text = res_full 
    tokens = None

    return res_text, tokens

def get_embeddings(embed_model, text_vals):
    # Gets text embeddings
    # replace newlines, which can negatively affect performance.
    text_vals_mod = [text.replace("\n", " ") for text in text_vals]

    if embed_model.batch_size is not None:
        # Run batched version and avoid hitting maximum embedding batch size.
        num_texts = len(text_vals_mod)
        batch_size = embed_model.batch_size
        batched_text_vals = np.array_split(text_vals_mod, np.arange(
            batch_size, num_texts, batch_size))
        embeddings = []
        for batch_text_vals in batched_text_vals:
            batch_embeddings = embed_model.fn(embed_model, batch_text_vals)
            embeddings += batch_embeddings
    else:
        # Run non-batched version
        embeddings = []
        for text_val in text_vals_mod:
            embedding = embed_model.fn(embed_model, text_val)
            embeddings.append(embedding)
    return np.array(embeddings)
