# OpenAI custom functions
import tiktoken
import numpy as np

# SETUP functions
def setup_llm_fn(api_key):
    from openai import AsyncOpenAI
    
    llm_client = AsyncOpenAI(
        api_key=api_key,
    )
    return llm_client

def setup_embed_fn(api_key):
    from openai import OpenAI
    
    embed_client = OpenAI(
        api_key=api_key,
    )
    return embed_client


# MODEL CALL functions
async def call_llm_fn(model, prompt):
    if "system_prompt" not in model.args:
        model.args["system_prompt"] = "You are a helpful assistant who helps with identifying patterns in text examples."
    if "temperature" not in model.args:
        model.args["temperature"] = 0

    # Preprocessing (custom to OpenAI setup)
    prompt = model.truncate_fn(model, prompt, out_token_alloc=1500)
        
    res = await model.client.chat.completions.create(
        model=model.name,
        temperature=model.args["temperature"],
        messages=[
            {"role": "system", "content": model.args["system_prompt"]},
            {"role": "user", "content": prompt},
        ]
    )
    res_parsed = res.choices[0].message.content if res else None
    in_tokens = (res.usage.prompt_tokens) if res is not None else 0
    out_tokens = (res.usage.completion_tokens) if res is not None else 0
    tokens = (in_tokens, out_tokens)
    return res_parsed, tokens

def call_embed_fn(model, texts_arr):
    resp = model.client.embeddings.create(
        input=texts_arr,
        model=model.name,
    )
    embeddings = [r.embedding for r in resp.data]
    tokens = np.sum(model.count_tokens_fn(model, text) for text in texts_arr)
    return embeddings, tokens


# TOKEN + COST functions
def count_tokens_fn(model, text):
    # Fetch the number of tokens used by the provided text
    encoding = tiktoken.encoding_for_model(model.name)
    tokens = encoding.encode(text)
    n_tokens = len(tokens)
    return n_tokens

def cost_fn(model, tokens):
    # Calculate cost with the tokens and provided model
    if model.cost is None:
        return None
    in_tokens, out_tokens = tokens
    in_cost, out_cost = model.cost
    in_total = in_tokens * in_cost
    out_total = out_tokens * out_cost
    return in_total, out_total

def truncate_tokens_fn(model, text, out_token_alloc=1500):
    encoding = tiktoken.encoding_for_model(model.name)
    tokens = encoding.encode(text)
    n_tokens = len(tokens)

    max_tokens = model.context_window - out_token_alloc
    if n_tokens > max_tokens:
        # Truncate the prompt
        tokens = tokens[:max_tokens]
        n_tokens = max_tokens
    out_text = encoding.decode(tokens)
    return out_text


# MODEL INFO functions (to fetch default values for subset of OpenAI models)
def get_context_window(model_name):
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name]["context_window"]
    raise Exception(f"Model {model_name} not in our defaults. Please specify the `context_window` parameter within the OpenAIModel instance. See https://platform.openai.com/docs/models for more info.")

def get_cost(model_name):
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name]["cost"]
    raise Exception(f"Model {model_name} not in our defaults. Please specify the `cost` parameter within the OpenAIModel instance in the form: (input_cost_per_token, output_cost_per_token). See https://openai.com/pricing for more info.")

def get_rate_limit(model_name):
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name]["rate_limit"]
    raise Exception(f"Model {model_name} not in our defaults. Please specify the `rate_limit` parameter within the OpenAIModel instance in the form: (n_requests, wait_time_secs). See https://platform.openai.com/account/limits to inform rate limit choices.")

# Model info: https://platform.openai.com/docs/models
# Pricing: https://openai.com/pricing
# Account rate limits: https://platform.openai.com/account/limits
TOKENS_1M = 1_000_000
MODEL_INFO = {
    # Format:
    # "model_name": {
    #     "context_window": <n_tokens>,
    #     "cost": (input_cost, output_cost),
    #     "rate_limit": (n_requests, wait_time_secs)
    # },
    "gpt-3.5-turbo": {
        "context_window": 16385,
        "cost": (1/TOKENS_1M, 2/TOKENS_1M),
        "rate_limit": (300, 10),  # = 300*6 = 1800 rpm
    },
    "gpt-4": {
        "context_window": 8192,
        "cost": (30/TOKENS_1M, 60/TOKENS_1M),
        "rate_limit": (20, 10),  # = 20*6 = 120 rpm
    },
    "gpt-4-turbo-preview": {
        "context_window": 128000,
        "cost": (10/TOKENS_1M, 30/TOKENS_1M),
        "rate_limit": (20, 10),  # = 20*6 = 120 rpm
    },
    "gpt-4-turbo": {
        "context_window": 128000,
        "cost": (10/TOKENS_1M, 30/TOKENS_1M),
        "rate_limit": (20, 10),  # = 20*6 = 120 rpm
    },
    "gpt-4o": {
        "context_window": 128000,
        "cost": (5/TOKENS_1M, 15/TOKENS_1M),
        "rate_limit": (20, 10)  # = 20*6 = 120 rpm
    },
    "gpt-4o-mini": {
        "context_window": 128000,
        "cost": (0.15/TOKENS_1M, 0.6/TOKENS_1M),
        "rate_limit": (300, 10)  # = 300*6 = 1800 rpm
    },
    "text-embedding-ada-002":{
        "cost": (0.1/TOKENS_1M, 0),
    },
    "text-embedding-3-small": {
        "cost": (0.02/TOKENS_1M, 0),
    },
    "text-embedding-3-large": {
        "cost": (0.13/TOKENS_1M, 0),
    },
}