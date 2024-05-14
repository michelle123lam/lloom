"""
llm.py
------
This file contains utility functions for processing calls to LLMs.
"""

# IMPORTS ================================
import time
import random
from pathos.multiprocessing import Pool
import hashlib
import numpy as np
import os

import asyncio
import tiktoken

# OPENAI setup =============================
import openai
from openai import OpenAI, AsyncOpenAI

if "OPENAI_API_KEY" not in os.environ:
    raise Exception("API key not found. Please set the OPENAI_API_KEY environment variable by running: `os.environ['OPENAI_API_KEY'] = 'your_key'`")
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
embed_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# CONSTANTS ================================
SYS_TEMPLATE = "You are a helpful assistant who helps with identifying patterns in text examples."

RATE_LIMITS = {
    # https://platform.openai.com/account/limits
    # (n_requests, wait_time_secs)
    "gpt-3.5-turbo": (300, 10), # = 300*6 = 1800 rpm
    "gpt-4": (20, 10), # = 20*6 = 120 rpm
    "gpt-4-turbo-preview": (20, 10), # = 20*6 = 120 rpm
    "gpt-4-turbo": (20, 10), # = 20*6 = 120 rpm
}

CONTEXT_WINDOW = {
    # https://platform.openai.com/docs/models
    # Total tokens shared between input and output
    "gpt-3.5-turbo": 16385,  # Max 4096 output tokens
    "gpt-4": 8192, 
    "gpt-4-turbo-preview": 128000,  # Max 4096 output tokens
    "gpt-4-turbo": 128000,  # Max 4096 output tokens
}

COSTS = {
    # https://openai.com/pricing
    "gpt-3.5-turbo": [0.0005/1000, 0.0015/1000],
    "gpt-4": [0.03/1000, 0.06/1000],
    "gpt-4-turbo-preview": [0.01/1000, 0.03/1000],
    "gpt-4-turbo": [0.01/1000, 0.03/1000],
}

EMBED_COSTS = {
    "text-embedding-ada-002": (0.00010/1000),
    "text-embedding-3-small": (0.00002/1000),
    "text-embedding-3-large": (0.00013/1000),
}

def get_token_estimate(text, model_name):
    # Fetch the number of tokens used by a prompt
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    return num_tokens

def get_token_estimate_list(text_list, model_name):
    # Fetch the number of tokens used by a list of prompts
    token_list = [get_token_estimate(text, model_name) for text in text_list]
    return np.sum(token_list)

def truncate_text_tokens(text, model_name, max_tokens):
    # Truncate a prompt to fit within a maximum number of tokens
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    n_tokens = len(tokens)
    if max_tokens is not None and n_tokens > max_tokens:
        # Truncate the prompt
        tokens = tokens[:max_tokens]
        n_tokens = max_tokens
    text = encoding.decode(tokens)
    return text, n_tokens

def calc_cost_by_tokens(model_name, in_tokens, out_tokens):
    # Calculate cost with the tokens and model name
    in_cost = in_tokens * COSTS[model_name][0]
    out_cost = out_tokens * COSTS[model_name][1]
    return in_cost, out_cost

# RETRYING + MULTIPROCESSING ================================

""" 
Defines a Python decorator to avoid API rate limits by retrying with exponential backoff.
From: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
"""
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                print("DELAY ", delay)
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                print("Exception", e)

    return wrapper

# CUSTOM LLM API WRAPPERS ================================

async def base_api_wrapper(cur_prompt, model_name, temperature):
    # Wrapper for calling the base OpenAI API
    cur_prompt = truncate_prompt(cur_prompt, model_name, out_token_alloc=1500)
    res = await client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYS_TEMPLATE},
            {"role": "user", "content": cur_prompt},
        ]
    )
    return res

def get_prompt_hash(p):
    user_message = p[1].content  # Isolate the user message
    hash = hashlib.sha256(user_message.encode()).hexdigest()
    return hash

def truncate_prompt(prompt, model_name, out_token_alloc):
    # Truncate a prompt to fit within a maximum number of tokens
    max_tokens = CONTEXT_WINDOW[model_name] - out_token_alloc
    prompt, n_tokens = truncate_text_tokens(prompt, model_name, max_tokens)
    return prompt

# Internal function making calls to LLM; runs a single LLM query
async def multi_query_gpt(model_name, prompt_template, arg_dict, batch_num=None, wait_time=None, temperature=0, debug=False):
    if wait_time is not None:
        if debug:
            print(f"Batch {batch_num}, wait time {wait_time}")
        await asyncio.sleep(wait_time)  # wait asynchronously

    try: 
        prompt = prompt_template.format(**arg_dict)
        res = await base_api_wrapper(prompt, model_name, temperature)
    except Exception as e:
        print("Error", e)
        return None
    
    return res

def get_res_str(res):
    # Fetch the response string OpenAI response JSON
    return res.choices[0].message.content

def process_results(results):
    # Extract just the text generations from response JSONs
    # Insert None if the result is None
    res_text = [(get_res_str(res) if res else None) for res in results]
    return res_text

async def multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name, rate_limits=None, temperature=0, batch_num=None, batched=True, debug=False):
    # Run multiple LLM queries
    if debug:
        print("model_name", model_name)
    
    if rate_limits is None:
        rate_limits = RATE_LIMITS  # Use default values

    if not batched:
        # Non-batched version
        tasks = [multi_query_gpt(model_name, prompt_template, args, temperature=temperature) for args in arg_dicts]
    else:
        # Batched version
        n_requests, wait_time_secs = rate_limits[model_name]
        tasks = []
        arg_dict_batches = [arg_dicts[i:i + n_requests] for i in range(0, len(arg_dicts), n_requests)]
        for inner_batch_num, cur_arg_dicts in enumerate(arg_dict_batches):
            if batch_num is None:
                wait_time = wait_time_secs * inner_batch_num
            else:
                wait_time = wait_time_secs * batch_num
            if debug:
                wait_time = 0 # Debug mode
            cur_tasks = [multi_query_gpt(model_name, prompt_template, arg_dict=args, batch_num=batch_num, wait_time=wait_time, temperature=temperature) for args in cur_arg_dicts]
            tasks.extend(cur_tasks)

    res_full = await asyncio.gather(*tasks)

    res_text = process_results(res_full)
    return res_text, res_full

def get_embeddings(embed_model_name, text_vals):
    # Gets OpenAI embeddings
    # replace newlines, which can negatively affect performance.
    text_vals_mod = [text.replace("\n", " ") for text in text_vals]
    resp = embed_client.embeddings.create(
        input=text_vals_mod,
        model=embed_model_name,
    )
    embeddings = [r.embedding for r in resp.data]
    return np.array(embeddings)
