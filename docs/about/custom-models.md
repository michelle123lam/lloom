# Custom Models

LLooM supports custom models to serve the Distill, Cluster, Synthesize, and Score operators, including (1) [different OpenAI models](./custom-models#different-openai-models), (2) [alternative LLM APIs](./custom-models#alternative-llm-apis) (e.g., Gemini), and (3) [open source models](./custom-models#open-source-models) (e.g., Llama). 

In all of these cases, users provide their own model specification when **creating their LLooM instance** (optionally with a short API-specific function implementation). All other steps remain the same as shown on the [Getting Started](./get-started.md) page.

## Different OpenAI Models
See our example [Colab notebook](https://colab.research.google.com/drive/1GKk7my5QA8rs7V4pi-WP0vJIvBYEkPnZ?usp=sharing) or follow these steps to specify different OpenAI models for the LLooM operators. 

First, import our helper classes for the OpenAI models:
```py
from text_lloom.llm import OpenAIModel, OpenAIEmbedModel
```

Then, prep your OpenAI API key to provide in your model spec:
```py
openai_key = "sk-YOUR-KEY-HERE"
```

Then, specify the desired model name and API key for as many operators as you would like (among `distill_model`, `cluster_model`, `synth_model`, and `score_model`). If any of these are excluded, the system will use the default settings, as described on the Getting Started page.

```py {5-30}
l = wb.lloom(
    df=df,
    id_col="doc_id",
    text_col="text",
    # Example custom OpenAI model spec  
    distill_model=OpenAIModel(  
        name="gpt-4o-mini",  
        api_key=openai_key,  
    ),  
    cluster_model=OpenAIEmbedModel(  
        name="text-embedding-3-large",  
        api_key=openai_key,  
    ),  
    synth_model=OpenAIModel(  
        name="gpt-4o",  
        api_key=openai_key,  
    ),  
    score_model=OpenAIModel(  
        name="gpt-4o-mini",  
        api_key=openai_key,  
    ),  
)
```

## Different rate limits
Beyond specifying different model names, you can also modify your desired **rate limit** based on dataset size and API limits. This parameter is set to a tuple `(n_requests, wait_time_secs)`:
- `n_requests`: Number of requests allowed in one batch.
- `wait_time_secs`: Time period (in seconds) to wait after a batch before making more requests.

This means that RPM (requests per minute) = `n_requests * (60 / wait_time_secs)`. For example, a rate limit tuple of `(40, 10)` specifies 40 requests every 10 seconds, which means 240 requests per minute.

You may also modify information about the context window and cost if these parameters become out of date.

```py
l = wb.lloom(
    df=df,
    id_col="doc_id",
    text_col="text",
    # Example custom OpenAI model spec
    distill_model=OpenAIModel(
        name="gpt-4o-mini",
        api_key=openai_key,
        context_window=128_000, # <n_tokens>  # [!code ++]
        cost=(0.15 / 1_000_000, 0.6 / 1_000_000), # (input_cost, output_cost)  # [!code ++]
        rate_limit=(300, 10), # (n_requests, wait_time_secs)  # [!code ++]
    ),
    cluster_model=OpenAIEmbedModel(
        name="text-embedding-3-large",
        api_key=openai_key,
    ),
    synth_model=OpenAIModel(
        name="gpt-4o",
        api_key=openai_key,
        context_window=128_000, # <n_tokens>  # [!code ++]
        cost=(2.5 / 1_000_000, 10 / 1_000_000), # (input_cost, output_cost) # [!code ++]
        rate_limit=(20, 10), # (n_requests, wait_time_secs)  # [!code ++]
    ),
    score_model=OpenAIModel(
        name="gpt-4o-mini",
        api_key=openai_key,
        context_window=128_000, # <n_tokens>  # [!code ++]
        cost=(0.15/1_000_000, 0.6/1_000_000), # (input_cost, output_cost) # [!code ++]
        rate_limit=(300, 10), # (n_requests, wait_time_secs)  # [!code ++]
    ),
)
```

## Alternative LLM APIs
See our example [Colab notebook](https://colab.research.google.com/drive/1uY1JcLA_3Bu7C34Pb9S_qLtbADIDjc5j?usp=sharing) to follow along on how to incorporate different LLM APIs to serve LLooM operators.

First, import our helper classes for custom models:
```py
from text_lloom.llm import Model, EmbedModel
```

Then, the key difference for non-OpenAI APIs is that, for new models, you must implement two functions to (1) perform any necessary API **setup** operations and (2) to **call** the provided API and process its outputs. We will demonstrate these using the Gemini API.

### Setup functions
A **setup** function takes an `api_key` as a parameter and is expected to return a `client` object, which is required for many LLM APIs and is used in the **call** functions. If the API does not use a client, this can be set as `None`. The function should perform any imports or initializations that are needed for the API to work properly.

#### Ex: Setup functions for Gemini
```py
# Setup for **LLM** API
# (relevant for Distill, Synthesize, Score operators)
def setup_llm_fn(api_key):
    import google.generativeai as genai  # Import package
    genai.configure(api_key=api_key)  # Set API key
    llm_client = None  # No client, so set to None
    return llm_client

# Setup for **Text Embedding** API
# (relevant for Cluster operator)
def setup_embed_fn(api_key):
    import google.generativeai as genai  # Import package
    genai.configure(api_key=api_key)  # Set API key
    embed_client = None  # No client, so set to None
    return embed_client
```

### Call function (LLM API)
A **call** function for an **LLM API** is an `async` function that takes in two arguments: 
- `model`: An instance of LLooM's custom `Model` object, which stores the specified model name, the API client, and other necessary parameters.
- `prompt`: The LLM prompt.

The function is expected to return two things:
- `text`: The LLM text response to the provided prompt.
- `tokens`: A `(in_tokens, out_tokens)` tuple of the tokens used to respond to this request. If not provided by the API, this can be set to `(0, 0)`, but this will prevent the system from accurately accounting for the token usage in cost tracking functions.

#### Ex: Call function for Gemini LLM API
```py
async def call_llm_fn(model, prompt):
    # Retrieve model response
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    model = genai.GenerativeModel(model.name)
    response = model.generate_content(prompt)
    text_result = response.text

    # Retrieve token usage
    in_tokens = response.usage_metadata.prompt_token_count
    out_tokens = response.usage_metadata.total_token_count
    tokens = (in_tokens, out_tokens)

    return response.text, tokens
```

### Call function (Text Embedding API)
Similarly, a **call** function for a **Text Embedding API** takes in two arguments: 
- `model`: An instance of LLooM's custom `EmbedModel` object, which stores the specified model name, the API client, and other necessary parameters.
- `text`: The text input for embedding retrieval.

The function is expected to return two things:
- `embedding`: The generated embedding for the provided text.
- `tokens`: A `(in_tokens, out_tokens)` tuple of the tokens used to respond to this request. Again, if not provided by the API, this can be set to `(0, 0)`.

#### Ex: Call function for Gemini Text Embedding API
```py
def call_embed_fn(model, text):
    task_type = "clustering"
    result = genai.embed_content(
        model=model.name,
        content=text,
        task_type=task_type,
    )
    tokens = [0, 0]
    return result['embedding'], tokens
```

### Custom LLooM Instance
Putting these together, we can now write a model specification to create a LLooM instance. The setup and call functions can be written once and reused for *any* instance that uses the same LLM API.

#### Ex: Gemini LLooM instance
```py {5-25}
l = wb.lloom(
    df=df,
    id_col="doc_id",
    text_col="text",
    # Example custom LLM API model spec
    distill_model=Model(
        setup_fn=setup_llm_fn,
        fn=call_llm_fn,
        name="gemini-1.5-flash", cost=[0.0005/1000, 0.0015/1000], rate_limit=(300, 10), context_window=16385, api_key=api_key
    ),
    cluster_model=EmbedModel(
        setup_fn=setup_embed_fn,
        fn=call_embed_fn,
        name="models/embedding-001", cost=(0.00013/1000), batch_size=2048, api_key=api_key
    ),
    synth_model=Model(
        setup_fn=setup_llm_fn,
        fn=call_llm_fn,
        name="gemini-1.5-flash", cost=[0.01/1000, 0.03/1000], rate_limit=(20, 10), context_window=128000, api_key=api_key
    ),
    score_model=Model(
        setup_fn=setup_llm_fn,
        fn=call_llm_fn,
        name="gemini-1.5-flash", cost=[0.0005/1000, 0.0015/1000], rate_limit=(300, 10), context_window=16385, api_key=api_key
    ),
)
```

## Open Source Models 
We can use the same approach generated above in the **Alternative LLM APIs** section to support open source models with the help of [vLLM](https://github.com/vllm-project/vllm).

This setup requires a GPU to host the open source model. First, follow the steps to set up vLLM on this machine and start the local vLLM API server (see [vLLM documentation](https://docs.vllm.ai/en/latest/)). After this, you can implement the [setup functions](./custom-models#setup-functions) and [call functions](./custom-models#call-function-llm-api) for your models and create a custom LLooM instance using the same method shown in the last section.

### Ex: Llama & Sentence Transformers
In this example, we will use `Meta-Llama-3-8B-Instruct` for our LLM. Start the local vLLM API server (for example, on `localhost:8000`):
```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --download-dir <path_to_your_hfcache_dir_here>
```

From a notebook, you can optionally test that this server is running:
```py
from openai import OpenAI  # Not using OpenAI API, but required for using vLLM's OpenAI-compatible API
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct",
                                      prompt="San Francisco is a")
print("Completion result:", completion)
```

Then, you can implement the setup and call functions for your model:
```py
import text_lloom.workbench as wb
from text_lloom.llm import Model, EmbedModel

# SETUP functions
# Meta-Llama-3-8B-Instruct for LLM
def setup_llm_fn(api_key):
    from openai import OpenAI  # Not using directly; just for vLLM's OpenAI-compatible API
    openai_api_base = "http://localhost:8000/v1"
    llm_client = OpenAI(
        api_key=api_key,
        base_url=openai_api_base,
    )
    return llm_client

# all-MiniLM-L6-v2 Sentence Transformers model for text embeddings
def setup_embed_fn(api_key):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

# CALL functions
async def call_llm_fn(model, prompt):
    if "system_prompt" not in model.args:
        model.args["system_prompt"] = "You are a helpful assistant who helps with identifying patterns in text examples."
    if "temperature" not in model.args:
        model.args["temperature"] = 0
        
    res = model.client.chat.completions.create(
        model=model.name,
        temperature=model.args["temperature"],
        messages=[
            {"role": "system", "content": model.args["system_prompt"]},
            {"role": "user", "content": prompt},
        ]
    )
    res_parsed = res.choices[0].message.content if res else None
    tokens = [0, 0]
    return res_parsed, tokens

def call_embed_fn(model, text_arr):
    embed_model = model.client
    embeddings = embed_model.encode(text_arr)
    embeddings = embeddings.tolist()
    tokens = [0, 0]
    return embeddings, tokens

api_key = "EMPTY"
```

Finally, you can create your LLooM instance using the open-source models:
```py {5-25}
l = wb.lloom(
    df=df,
    id_col="doc_id",
    text_col="text",
    # Open source model spec
    distill_model=Model(
        setup_fn=setup_llm_fn, 
        fn=call_llm_fn, 
        name="meta-llama/Meta-Llama-3-8B-Instruct", cost=[0,0], rate_limit=(300, 10), context_window=16385, api_key=api_key
    ),
    cluster_model=EmbedModel(
        setup_fn=setup_embed_fn, 
        fn=call_embed_fn, 
        name="", cost=(0), batch_size=2048, api_key=api_key
    ),
    synth_model=Model(
        setup_fn=setup_llm_fn, 
        fn=call_llm_fn, 
        name="meta-llama/Meta-Llama-3-8B-Instruct", cost=[0,0], rate_limit=(20, 10), context_window=128000, api_key=api_key
    ),
    score_model=Model(
        setup_fn=setup_llm_fn, 
        fn=call_llm_fn, 
        name="meta-llama/Meta-Llama-3-8B-Instruct", cost=[0,0], rate_limit=(300, 10), context_window=16385, api_key=api_key
    ),
)
```
