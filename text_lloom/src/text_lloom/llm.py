"""
llm.py
------
This file contains utility functions for processing calls to LLMs.
"""

# IMPORTS ================================
import numpy as np
import asyncio
import llm_openai as llm_openai

# MODEL CLASSES ================================
class Model:
    # Specification to run LLooM operator with a specific large language model
    # - name: str, name of the model (ex: "gpt-3.5-turbo")
    # - setup_fn: function, function to set up the LLM client
    # - fn: function, function to call the model (i.e., to run LLM prompt)
    # - api_key: str (optional), API key
    # - rate_limit: tuple (optional), (n_requests, wait_time_secs)
    def __init__(self, name, setup_fn, fn, api_key=None, rate_limit=None, **args):
        self.name = name
        self.setup_fn = setup_fn
        self.fn = fn
        self.rate_limit = rate_limit
        self.client = setup_fn(api_key)
        self.args = args


class EmbedModel:
    # Specification to run LLooM operator with a specific embedding model
    # - name: str, name of the model (ex: "text-embedding-ada-002")
    # - setup_fn: function, function to set up the embedding client
    # - fn: function, function to call the model (i.e., to fetch embedding)
    # - api_key: str (optional), API key
    # - batch_size: int (optional), maximum batch size for embeddings (None for no batching)
    def __init__(self, name, setup_fn, fn, api_key=None, batch_size=None, **args):
        self.name = name
        self.setup_fn = setup_fn
        self.fn = fn
        self.batch_size = batch_size
        self.client = setup_fn(api_key)
        self.args = args

class OpenAIModel(Model):
    # OpenAIModel class for OpenAI LLMs
    # Adds the following parameters for token and cost tracking:
    # - context_window: int (optional), total tokens shared between input and output
    # - cost: float (optional), cost per token (input_cost, output_cost)
    def __init__(self, name, api_key, setup_fn=llm_openai.setup_llm_fn, fn=llm_openai.call_llm_fn, rate_limit=None, context_window=None, cost=None, **args):
        super().__init__(name, setup_fn, fn, api_key, rate_limit, **args)
        # OpenAI-specific setup
        # TODO: add helpers to support token and cost tracking for other models
        self.truncate_fn = llm_openai.truncate_tokens_fn  # called in llm_openai.py call_llm_fn()
        self.cost_fn = llm_openai.cost_fn  # called in concept_induction.py calc_cost()
        self.count_tokens_fn = llm_openai.count_tokens_fn  # called in workbench.py estimate_gen_cost()

        if context_window is None:
            context_window = llm_openai.get_context_window(name)
        self.context_window = context_window
        
        if cost is None:
            cost = llm_openai.get_cost(name)
        self.cost = cost

        if rate_limit is None:
            rate_limit = llm_openai.get_rate_limit(name)
        self.rate_limit = rate_limit
        

class OpenAIEmbedModel(EmbedModel):
    # OpenAIEmbedModel class for OpenAI embedding models
    # Adds the following parameters for cost tracking:
    # - cost: float (optional), cost per token (input_cost, output_cost)
    def __init__(self, name, setup_fn=llm_openai.setup_embed_fn, fn=llm_openai.call_embed_fn, api_key=None, batch_size=2048, cost=None, **args):
        super().__init__(name, setup_fn, fn, api_key, batch_size, **args)
        # OpenAI-specific setup
        self.count_tokens_fn = llm_openai.count_tokens_fn  # called in llm_openai.py call_embed_fn()
        if cost is None:
            cost = llm_openai.get_cost(name)
        self.cost = cost

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
        return None, None  # result, tokens
    
    return res

# Run multiple LLM queries
async def multi_query_gpt_wrapper(prompt_template, arg_dicts, model, batch_num=None, batched=True, debug=False):
    if debug:
        print("model_name", model.name)
    
    rate_limit = model.rate_limit
    if (not batched) or (rate_limit is None):
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

    # Unpack results into the text and token counts
    res_text, tokens_list = list(zip(*res_full))
    in_tokens = np.sum([tokens[0] for tokens in tokens_list if tokens is not None])
    out_tokens = np.sum([tokens[1] for tokens in tokens_list if tokens is not None])
    tokens = (in_tokens, out_tokens)
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
        token_counts = []
        for batch_text_vals in batched_text_vals:
            batch_embeddings, tokens = embed_model.fn(embed_model, batch_text_vals)
            embeddings += batch_embeddings
            token_counts.append(tokens)
    else:
        # Run non-batched version
        embeddings = []
        token_counts = []
        for text_val in text_vals_mod:
            embedding, tokens = embed_model.fn(embed_model, text_val)
            embeddings.append(embedding)
            token_counts.append(tokens)
    
    tokens = np.sum([count for count in token_counts if count is not None])
    return np.array(embeddings), tokens
