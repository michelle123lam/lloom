"""
llm.py
------
This file contains utility functions for processing calls to LLMs.
"""

# IMPORTS ================================
import openai
import google.generativeai as palm

import time
import random
from pathos.multiprocessing import Pool
import hashlib
import numpy as np

import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import tiktoken

# CONSTANTS ================================
SYS_TEMPLATE = "You are a helpful assistant who helps with identifying patterns in text examples."

RATE_LIMITS = {
    # https://platform.openai.com/account/limits
    # (n_requests, wait_time_secs)
    "gpt-3.5-turbo": (300, 10), # = 300*6 = 1800 rpm (max 10k requests per minute for org)
    "gpt-4": (20, 10), # = 20*6 = 120 rpm (max 10k requests per minute for org)
    "gpt-4-turbo-preview": (20, 10), # = 20*6 = 120 rpm, testing purposes
    "palm": (9, 10), # = 9*6 = 54 rpm (max 90 requests per minute)
}

CONTEXT_WINDOW = {
    # https://platform.openai.com/docs/models
    # Total tokens shared between input and output
    "gpt-3.5-turbo": 16385,  # Max 4096 output tokens
    "gpt-4": 8192, 
    "gpt-4-turbo-preview": 128000,  # Max 4096 output tokens
    "palm": 8000,
}

COSTS = {
    # https://openai.com/pricing
    "gpt-3.5-turbo": [0.0005/1000, 0.0015/1000],
    "gpt-4": [0.03/1000, 0.06/1000],
    "gpt-4-turbo-preview": [0.01/1000, 0.03/1000],
}

EMBED_COSTS = {
    "text-embedding-ada-002": (0.00010/1000),
    "text-embedding-3-small": (0.00002/1000),
    "text-embedding-3-large": (0.00013/1000),
}

def get_system_prompt():
    system_message_prompt = SystemMessagePromptTemplate.from_template(SYS_TEMPLATE)
    return system_message_prompt

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
    errors: tuple = (openai.error.RateLimitError,),
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
def get_res_str(res):
    # Fetch the response string from an LLM response JSON
    return res["choices"][0]["message"]["content"]

@retry_with_exponential_backoff
def base_api_wrapper(cur_prompt, model_name="gpt-3.5-turbo"):
    # Wrapper for calling the base OpenAI API
    res = openai.ChatCompletion.create(
        model=model_name,
        messages=[
                {"role": "system", "content": SYS_TEMPLATE},
                {"role": "user", "content": cur_prompt},
            ]
    )
    res_str = get_res_str(res)
    return res_str

@retry_with_exponential_backoff
def base_api_wrapper_palm(cur_prompt):
    # Uses the chat model (chat-bison-001)
    response = palm.chat(messages=cur_prompt)
    res_str = response.last
    return res_str

""" 
Handles multiprocessing of multiple model queries

Inputs: 
    - prompts: list of prompts to execute
    - multiprocessing: bool. Run multiprocessing if True, otherwise run sequentially

Outputs: List of outputs generated by all models
"""
def multi_query(prompts, model_name, multiprocessing=True):
    if multiprocessing: 
        if model_name != "palm":
            # GPT
            pool = Pool(processes=len(prompts))
            results = pool.map(base_api_wrapper, prompts) # unpacks prompt and model name
            pool.close()
        elif model_name == "palm":
            # PaLM
            batch_size = 90
            delay = 60
            # Run only `batch_size` prompts at a time
            for i in range(0, len(prompts), batch_size):
                cur_prompts = prompts[i:i+batch_size]
                pool = Pool(processes=len(cur_prompts))
                results = pool.map(base_api_wrapper_palm, prompts) # unpacks prompt and model name
                pool.close()
                time.sleep(delay)

    else:
        results = []
        for p, model_name in prompts:
            r = base_api_wrapper(p, model_name)
            results.append(r)
    
    return results

# Mock classes for testing ====================================
class LangChainGeneration():
    # Mock class for a generation within a LangChainResult
    def __init__(self, response):
        self.text = response

class LangChainResult:
    # Mock class for a result from a LangChain API call
    def __init__(self, response):
        gen = LangChainGeneration(response)
        self.generations = [[gen]]
        self.llm_output = {
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        }

async def get_test_response(example_ids):
    # Returns a dummy response for testing, currently only for the `themes_to_examples` prompt
    # TODO: Adapt to support other prompts
    full_template = """
    {{
        "pattern_results": [
            {results_str}
        ]
    }}
    """
    def get_ex_json(ex_id):
        template = """
            {{
                "example_id": "{example_id}",
                "rationale": "sample rationale",
                "answer": "A",
            }}
        """
        return template.format(example_id=ex_id)
    
    results = [get_ex_json(ex_id) for ex_id in example_ids]
    results_str = ",\n".join(results)
    res = LangChainResult(response=full_template.format(results_str=results_str))
    return res

def get_prompt_hash(p):
    user_message = p[1].content  # Isolate the user message
    hash = hashlib.sha256(user_message.encode()).hexdigest()
    return hash

def truncate_prompt(prompt, model_name, out_token_alloc):
    # Truncate a prompt to fit within a maximum number of tokens
    max_tokens = CONTEXT_WINDOW[model_name] - out_token_alloc
    for i, message in enumerate(prompt):
        trunc_message, n_tokens = truncate_text_tokens(message.content, model_name, max_tokens)
        prompt[i].content = trunc_message
        max_tokens -= n_tokens
    return prompt

# Main function making calls to LLM
async def multi_query_gpt(chat_model, model_name, prompt_template, arg_dict, batch_num=None, wait_time=None, cache=False, debug=False):
    # Run a single query using LangChain OpenAI Chat API

    # System
    system_message_prompt = get_system_prompt()
    # Human
    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    # Execute prompt
    if wait_time is not None:
        if debug:
            print(f"Batch {batch_num}, wait time {wait_time}")
        await asyncio.sleep(wait_time)  # wait asynchronously
    chat_prompt_formatted = chat_prompt.format_prompt(**arg_dict).to_messages()
    chat_prompt_formatted = truncate_prompt(chat_prompt_formatted, model_name, out_token_alloc=1500)
    prompt_hash = get_prompt_hash(chat_prompt_formatted)

    # Run LLM generation
    try: 
        res_full = await chat_model.agenerate([chat_prompt_formatted])
    except Exception as e:
        print("Error", e)
        return None
    
    # TODO: Add back caching
    return res_full

def process_results(results):
    # Extract just the text generation from the LangChain OpenAI Chat results
    # Insert None if the result is None
    res_text = [(res.generations[0][0].text if res else None) for res in results]
    return res_text

async def multi_query_gpt_wrapper(prompt_template, arg_dicts, model_name, temperature=0, batch_num=None, batched=True, debug=False):
    # Multi-query using LangChain OpenAI Chat API
    chat_model = ChatOpenAI(temperature=temperature, model_name=model_name)
    if debug:
        print("model_name", model_name)
    

    if not batched:
        # Non-batched version
        tasks = [multi_query_gpt(chat_model, model_name, prompt_template, args) for args in arg_dicts]
    else:
        # Batched version
        n_requests, wait_time_secs = RATE_LIMITS[model_name]
        tasks = []
        arg_dict_batches = [arg_dicts[i:i + n_requests] for i in range(0, len(arg_dicts), n_requests)]
        for inner_batch_num, cur_arg_dicts in enumerate(arg_dict_batches):
            chat_model = ChatOpenAI(temperature=temperature, model_name=model_name)
            if batch_num is None:
                wait_time = wait_time_secs * inner_batch_num
            else:
                wait_time = wait_time_secs * batch_num
            if debug:
                wait_time = 0 # Debug mode
            cur_tasks = [multi_query_gpt(chat_model, model_name, prompt_template, arg_dict=args, batch_num=batch_num, wait_time=wait_time) for args in cur_arg_dicts]
            tasks.extend(cur_tasks)

    res_full = await asyncio.gather(*tasks)

    res_text = process_results(res_full)
    return res_text, res_full


# Currently unused
# def call_wrapper(sess, prompt_template, arg_dict, model_name=None, verbose=False):
#     # Wrapper for calling an LLM API, whether the OpenAI API (via base_api_wrapper) or other APIs like LangChain
#     res = None
#     if model_name is None:
#         model_name = sess.model
#     if sess.use_base_api:
#         if arg_dict is None:
#             cur_prompt = prompt_template
#         else:
#             cur_prompt = prompt_template.format(**arg_dict)
#         if verbose:
#             print(cur_prompt)
        
#         if model_name != "palm":
#             # GPT
#             resp = base_api_wrapper(cur_prompt, model_name)
#         elif model_name == "palm":
#             # PaLM
#             resp = base_api_wrapper_palm(cur_prompt)
#         res = resp
#     else:
#         raise Exception("Not implemented yet!")
    
#     # Save raw response to session's llm_cache
#     t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
#     sess.llm_cache[t] = res
#     return res
