# Distill - Filter ========================
filter_prompt = """
I have the following TEXT EXAMPLE:
{ex}

Please extract {n_quotes} QUOTES exactly copied from this EXAMPLE that are {seeding_phrase}. Please respond ONLY with a valid JSON in the following format:
{{
    "relevant_quotes": [ "<QUOTE_1>", "<QUOTE_2>", ... ]
}}
"""
# Removed: If there are no quotes relevant to {seed}, leave the list empty.

# Distill - Summarize ========================
summarize_prompt = """
I have the following TEXT EXAMPLE:
{ex}

Please summarize the main point of this EXAMPLE {seeding_phrase} into {n_bullets} bullet points, where each bullet point is a {n_words} word phrase. Please respond ONLY with a valid JSON in the following format:
{{
    "bullets": [ "<BULLET_1>", "<BULLET_2>", ... ]
}}
"""

# Synthesize ========================
synthesize_prompt = """
I have this set of bullet point summaries of text examples:
{examples}

Please write a summary of {n_concepts_phrase} for these examples. {seed_phrase} For each high-level pattern, write a 2-4 word NAME for the pattern and an associated 1-sentence ChatGPT PROMPT that could take in a new text example and determine whether the relevant pattern applies. Also include 1-2 example_ids for items that BEST exemplify the pattern. Please respond ONLY with a valid JSON in the following format:
{{
    "patterns": [ 
        {{"name": "<PATTERN_NAME_1>", "prompt": "<PATTERN_PROMPT_1>", "example_ids": ["<EXAMPLE_ID_1>", "<EXAMPLE_ID_2>"]}},
        {{"name": "<PATTERN_NAME_2>", "prompt": "<PATTERN_PROMPT_2>", "example_ids": ["<EXAMPLE_ID_1>", "<EXAMPLE_ID_2>"]}},
    ]
}}
"""

# Review ========================
review_remove_prompt = """
I have this set of themes generated from text examples:
{themes}

Please identify any themes that should be REMOVED because they are either:
(1) Too specific/narrow and would only describe a few examples, or 
(2) Too generic/broad and would describe nearly all examples.
If there no such themes, please leave the list empty.
Please respond ONLY with a valid JSON in the following format:

{{
    "remove": [ 
        "<THEME_NAME_5>",
        "<THEME_NAME_6>",
    ]
}}
"""

review_merge_prompt = """
I have this set of themes generated from text examples:
{themes}

Please identify any PAIRS of themes that are similar or overlapping that should be MERGED together. 
Please respond ONLY with a valid JSON in the following format with the original themes and a new name and prompt for the merged theme. Do NOT simply combine the prior theme names or prompts, but come up with a new 2-3 word name and 1-sentence ChatGPT prompt. If there no similar themes, please leave the list empty.

{{
    "merge": [ 
        {{
            "original_themes": ["<THEME_NAME_A>", "<THEME_NAME_B>"],
            "merged_theme_name": "<THEME_NAME_AB>",
            "merged_theme_prompt": "<THEME_PROMPT_AB>",
        }},
        {{
            "original_themes": ["<THEME_NAME_C>", "<THEME_NAME_D>"],
            "merged_theme_name": "<THEME_NAME_CD>",
            "merged_theme_prompt": "<THEME_PROMPT_CD>",
        }}
    ]
}}
"""

# Score ========================
score_no_highlight_prompt = """
CONTEXT: 
    I have the following text examples in a JSON:
    {examples_json}

    I also have a pattern named {pattern_name} with the following PROMPT: 
    {pattern_prompt}

TASK:
    For each example, please evaluate the PROMPT by generating a 1-sentence RATIONALE of your thought process and providing a resulting ANSWER of ONE of the following multiple-choice options, including just the letter: 
    - A: Strongly agree
    - B: Agree
    - C: Neither agree nor disagree
    - D: Disagree
    - E: Strongly disagree
    Respond with ONLY a JSON with the following format, escaping any quotes within strings with a backslash:
    {{
        "pattern_results": [
            {{
                "example_id": "<example_id>",
                "rationale": "<rationale>",
                "answer": "<answer>",
            }}
        ]
    }}
"""

score_highlight_prompt = """
CONTEXT: 
    I have the following text examples in a JSON:
    {examples_json}

    I also have a pattern named {pattern_name} with the following PROMPT: 
    {pattern_prompt}

TASK:
    For each example, please evaluate the PROMPT by generating a 1-sentence RATIONALE of your thought process and providing a resulting ANSWER of ONE of the following multiple-choice options, including just the letter: 
    - A: Strongly agree
    - B: Agree
    - C: Neither agree nor disagree
    - D: Disagree
    - E: Strongly disagree
    Please also include one 1-sentence QUOTE exactly copied from the example that illustrates this pattern.
    Respond with ONLY a JSON with the following format, escaping any quotes within strings with a backslash:
    {{
        "pattern_results": [
            {{
                "example_id": "<example_id>",
                "rationale": "<rationale>",
                "answer": "<answer>",
                "quote": "<quote>"
            }}
        ]
    }}
"""

score_overall_topic_prompt = """
CONTEXT: 
    I have the following text examples in a JSON:
    {examples_json}

    I also have a pattern named {pattern_name} with the following PROMPT: 
    AS ITS PRIMARY TOPIC, {pattern_prompt}

TASK:
    For each example, please evaluate the PROMPT by generating a 1-sentence RATIONALE of your thought process and providing a resulting ANSWER of ONE of the following multiple-choice options, including just the letter: 
    - A: Strongly agree
    - B: Agree
    - C: Neither agree nor disagree
    - D: Disagree
    - E: Strongly disagree
    Only answer with "A" if the example is PRIMARILY about the topic.
    Respond with ONLY a JSON with the following format, escaping any quotes within strings with a backslash:
    {{
        "pattern_results": [
            {{
                "example_id": "<example_id>",
                "rationale": "<rationale>",
                "answer": "<answer>",
            }}
        ]
    }}
"""

# Summarize Concept ========================
summarize_concept_prompt = """
Please write a BRIEF {summary_length} executive summary of the theme "{concept_name}" as it appears in the following examples.
{examples}

DO NOT write the summary as a third party using terms like "the text examples" or "they discuss", but write the summary from the perspective of the text authors making the points directly.
Please respond ONLY with a valid JSON in the following format:
{{
    "summary": "<SUMMARY>"
}}
"""

concept_auto_eval_prompt = """
I have this set of CONCEPTS:
{concepts}

I have this set of TEXTS: 
{items}

Please match at most ONE TEXT to each CONCEPT. To perform a match, the text must EXACTLY match the meaning of the concept. Do NOT match the same TEXT to multiple CONCEPTS.

Here are examples of VALID matches:
- Global Diplomacy, International Relations; rationale: "The text is about diplomacy between countries."
- Statistical Data, Quantitative Evidence; rationale: "The text is about data and quantitative measures."
- Policy and Regulation, Policy issues and legislation; rationale: "The text is about policy, laws, and legislation."

Here are examples of INVALID matches:
- Reputation Impact, Immigration
- Environment, Politics and Law
- Interdisciplinary Politics, Economy

If there are no valid matches, please EXCLUDE the concept from the list. Please provide a 1-sentence RATIONALE for your decision for any matches. 
Please respond with a list of each concept and either the item it matches or NONE if no item matches in this format:
{{
    "concept_matches": [
        {{
            "concept_id": "<concept_id_number>",
            "item_id": "<item_id_number or NONE>",
            "rationale": "<rationale for match>",
        }}
    ]
}}
"""
