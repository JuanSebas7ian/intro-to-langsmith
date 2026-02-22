#!/usr/bin/env python3
"""
Script to migrate intro-to-langsmith notebooks from OpenAI/ChatGPT to Amazon Bedrock.
Comments out OpenAI code and adds Bedrock equivalents.
"""
import json
import os
import re
import copy

BASE = "/home/juansebas7ian/intro-to-langsmith/notebooks"

NOVA2_LITE = "us.amazon.nova-2-lite-v1:0"
NOVA_MICRO = "amazon.nova-micro-v1:0"

def comment_and_replace_lines(lines, replacements):
    """
    For each (pattern, comment_replacement) in replacements:
    - Find lines matching pattern
    - Comment them out  
    - Add replacement lines after
    """
    new_lines = []
    for line in lines:
        matched = False
        for pattern, replacement_lines in replacements:
            if pattern in line and not line.strip().startswith("#"):
                # Comment original
                new_lines.append("# " + line)
                # Add replacements
                for r in replacement_lines:
                    new_lines.append(r + "\n")
                matched = True
                break
        if not matched:
            new_lines.append(line)
    return new_lines


def process_notebook(path, cell_processors):
    """Load notebook, apply processors to specific cells, save."""
    with open(path, 'r') as f:
        nb = json.load(f)
    
    code_cell_idx = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            if code_cell_idx in cell_processors:
                processor = cell_processors[code_cell_idx]
                source = cell['source']
                # Join, process, split back
                text = ''.join(source)
                new_text = processor(text)
                # Split back to lines preserving notebook format
                lines = []
                for i, line in enumerate(new_text.split('\n')):
                    if i < len(new_text.split('\n')) - 1:
                        lines.append(line + '\n')
                    else:
                        if line:  # don't add empty last line
                            lines.append(line)
                cell['source'] = lines
                # Clear outputs
                cell['outputs'] = []
                cell['execution_count'] = None
            code_cell_idx += 1
    
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  ✅ {os.path.basename(path)}")


def replace_env_cell(text):
    """Comment out OPENAI_API_KEY env var."""
    return text.replace(
        'os.environ["OPENAI_API_KEY"] = ""',
        '# os.environ["OPENAI_API_KEY"] = ""'
    )


def migrate_rag_app_cell(text):
    """Migrate the standard RAG app cell (used in module_0, module_1 tracing_basics)."""
    result = text
    # Comment imports
    result = result.replace(
        'from openai import OpenAI',
        '# from openai import OpenAI\nfrom langchain_aws import ChatBedrockConverse\nfrom langchain_core.messages import HumanMessage, SystemMessage'
    )
    # Comment provider/model
    result = result.replace(
        'MODEL_PROVIDER = "openai"',
        '# MODEL_PROVIDER = "openai"\nMODEL_PROVIDER = "bedrock"'
    )
    result = result.replace(
        'MODEL_NAME = "gpt-4o-mini"',
        f'# MODEL_NAME = "gpt-4o-mini"\nMODEL_NAME = "{NOVA2_LITE}"'
    )
    result = result.replace(
        'MODEL_NAME = "gpt-4o"',
        f'# MODEL_NAME = "gpt-4o"\nMODEL_NAME = "{NOVA2_LITE}"'
    )
    # Comment client
    result = result.replace(
        'openai_client = OpenAI()',
        '# openai_client = OpenAI()\nbedrock_client = ChatBedrockConverse(model=MODEL_NAME)'
    )
    # Comment call_openai function and replace
    result = result.replace(
        'return call_openai(messages)',
        '# return call_openai(messages)\n    return call_bedrock(messages)'
    )
    # Replace function definitions - handle different signatures
    if 'def call_openai(\n    messages: List[dict], model: str = MODEL_NAME, temperature: float = 0.0\n) -> str:\n    return openai_client.chat.completions.create(\n        model=model,\n        messages=messages,\n        temperature=temperature,\n    )' in result:
        result = result.replace(
            'def call_openai(\n    messages: List[dict], model: str = MODEL_NAME, temperature: float = 0.0\n) -> str:\n    return openai_client.chat.completions.create(\n        model=model,\n        messages=messages,\n        temperature=temperature,\n    )',
            'def call_openai(\n    messages: List[dict], model: str = MODEL_NAME, temperature: float = 0.0\n) -> str:\n    pass  # Comentado - usar call_bedrock\n    # return openai_client.chat.completions.create(\n    #     model=model,\n    #     messages=messages,\n    #     temperature=temperature,\n    # )\n\ndef call_bedrock(messages: list) -> str:\n    lc_messages = []\n    for m in messages:\n        if m["role"] == "system":\n            lc_messages.append(SystemMessage(content=m["content"]))\n        else:\n            lc_messages.append(HumanMessage(content=m["content"]))\n    return bedrock_client.invoke(lc_messages)'
        )
    # For simpler signature
    if 'def call_openai(\n    messages: List[dict],\n) -> str:\n    return openai_client.chat.completions.create(\n        model=MODEL_NAME,\n        messages=messages,\n    )' in result:
        result = result.replace(
            'def call_openai(\n    messages: List[dict],\n) -> str:\n    return openai_client.chat.completions.create(\n        model=MODEL_NAME,\n        messages=messages,\n    )',
            'def call_openai(\n    messages: List[dict],\n) -> str:\n    pass  # Comentado - usar call_bedrock\n    # return openai_client.chat.completions.create(\n    #     model=MODEL_NAME,\n    #     messages=messages,\n    # )\n\ndef call_bedrock(messages: list) -> str:\n    lc_messages = []\n    for m in messages:\n        if m["role"] == "system":\n            lc_messages.append(SystemMessage(content=m["content"]))\n        else:\n            lc_messages.append(HumanMessage(content=m["content"]))\n    return bedrock_client.invoke(lc_messages)'
        )
    # Replace response access
    result = result.replace(
        'return response.choices[0].message.content',
        '# return response.choices[0].message.content\n    return response.content'
    )
    # Comments in docstrings
    result = result.replace('call_openai', 'call_bedrock')
    result = result.replace('call_bedrock(messages)\n) -> str:\n    pass', 'call_openai(\n    messages: List[dict],\n) -> str:\n    pass')
    return result


def migrate_rag_app_cell_with_traceable(text):
    """Like migrate_rag_app_cell but for cells that already have @traceable."""
    result = migrate_rag_app_cell(text)
    return result


# ========== Process each notebook ==========

print("🔄 Migrating notebooks to Amazon Bedrock...\n")

# --- module_0/rag_application.ipynb ---
print("📁 module_0/")
path = f"{BASE}/module_0/rag_application.ipynb"
process_notebook(path, {
    0: replace_env_cell,  # env vars cell
    2: migrate_rag_app_cell,  # main RAG app cell
})

# --- module_1/tracing_basics.ipynb ---
print("\n📁 module_1/")
path = f"{BASE}/module_1/tracing_basics.ipynb"
process_notebook(path, {
    0: replace_env_cell,
    2: migrate_rag_app_cell,  # first RAG cell (without traceable)
    3: migrate_rag_app_cell_with_traceable,  # second RAG cell (with traceable + metadata)
})

# --- module_1/conversational_threads.ipynb ---
path = f"{BASE}/module_1/conversational_threads.ipynb"
def migrate_conv_threads(text):
    result = text
    result = result.replace(
        'from openai import OpenAI',
        '# from openai import OpenAI\nfrom langchain_aws import ChatBedrockConverse\nfrom langchain_core.messages import HumanMessage, SystemMessage'
    )
    result = result.replace(
        'openai_client = OpenAI()',
        '# openai_client = OpenAI()\nbedrock_client = ChatBedrockConverse(model="' + NOVA2_LITE + '")'
    )
    result = result.replace(
        'return call_openai(messages)',
        '# return call_openai(messages)\n    return call_bedrock(messages)'
    )
    # Replace the call_openai function
    result = result.replace(
        '@traceable(run_type="llm")\ndef call_openai(\n    messages: List[dict], model: str = "gpt-4o-mini", temperature: float = 0.0\n) -> str:\n    return openai_client.chat.completions.create(\n        model=model,\n        messages=messages,\n        temperature=temperature,\n    )',
        '# @traceable(run_type="llm")\n# def call_openai(\n#     messages: List[dict], model: str = "gpt-4o-mini", temperature: float = 0.0\n# ) -> str:\n#     return openai_client.chat.completions.create(\n#         model=model,\n#         messages=messages,\n#         temperature=temperature,\n#     )\n\n@traceable(run_type="llm")\ndef call_bedrock(messages: list) -> str:\n    lc_messages = []\n    for m in messages:\n        if m["role"] == "system":\n            lc_messages.append(SystemMessage(content=m["content"]))\n        else:\n            lc_messages.append(HumanMessage(content=m["content"]))\n    return bedrock_client.invoke(lc_messages)'
    )
    result = result.replace(
        'return response.choices[0].message.content',
        '# return response.choices[0].message.content\n    return response.content'
    )
    return result

process_notebook(path, {
    0: replace_env_cell,
    2: migrate_conv_threads,
})

# --- module_1/types_of_runs.ipynb ---
path = f"{BASE}/module_1/types_of_runs.ipynb"
def migrate_types_tool_cell(text):
    result = text
    result = result.replace(
        'from openai import OpenAI',
        '# from openai import OpenAI\nfrom langchain_aws import ChatBedrockConverse\nfrom langchain_core.messages import HumanMessage, SystemMessage'
    )
    result = result.replace(
        'openai_client = OpenAI()',
        '# openai_client = OpenAI()\nbedrock_client = ChatBedrockConverse(model="' + NOVA2_LITE + '")'
    )
    result = result.replace(
        '@traceable(run_type="llm")\ndef call_openai(\n    messages: List[dict], tools: Optional[List[dict]]\n) -> str:\n  return openai_client.chat.completions.create(\n    model="gpt-4o-mini",\n    messages=messages,\n    temperature=0,\n    tools=tools\n  )',
        '# @traceable(run_type="llm")\n# def call_openai(\n#     messages: List[dict], tools: Optional[List[dict]]\n# ) -> str:\n#   return openai_client.chat.completions.create(\n#     model="gpt-4o-mini",\n#     messages=messages,\n#     temperature=0,\n#     tools=tools\n#   )\n\n@traceable(run_type="llm")\ndef call_bedrock(\n    messages: List[dict], tools: Optional[List[dict]]\n) -> str:\n  lc_messages = []\n  for m in messages:\n      if m["role"] == "system":\n          lc_messages.append(SystemMessage(content=m["content"]))\n      elif m["role"] == "tool":\n          from langchain_core.messages import ToolMessage\n          lc_messages.append(ToolMessage(content=m["content"], tool_call_id=m.get("tool_call_id","")))\n      else:\n          lc_messages.append(HumanMessage(content=m["content"]))\n  kwargs = {}\n  if tools:\n      kwargs["tools"] = tools\n  return bedrock_client.invoke(lc_messages, **kwargs)'
    )
    result = result.replace(
        'response = call_openai(inputs, tools)',
        '# response = call_openai(inputs, tools)\n  response = call_bedrock(inputs, tools)'
    )
    result = result.replace(
        'output = call_openai(inputs, None)',
        '# output = call_openai(inputs, None)\n  output = call_bedrock(inputs, None)'
    )
    return result

process_notebook(path, {
    0: replace_env_cell,
    5: migrate_types_tool_cell,
})

# --- module_1/alternative_tracing_methods.ipynb ---
path = f"{BASE}/module_1/alternative_tracing_methods.ipynb"

def migrate_alt_langchain_cell(text):
    """Cell with ChatOpenAI (LangGraph)."""
    result = text
    result = result.replace(
        'from langchain_openai import ChatOpenAI',
        '# from langchain_openai import ChatOpenAI\nfrom langchain_aws import ChatBedrockConverse'
    )
    result = result.replace(
        'llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)',
        '# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)\nllm = ChatBedrockConverse(model="' + NOVA2_LITE + '")'
    )
    return result

def migrate_alt_context_cell(text):
    """Cell with trace context manager."""
    return migrate_rag_app_cell(text)

def migrate_alt_wrap_openai_cell(text):
    """Cell with wrap_openai."""
    result = text
    result = result.replace(
        'from langsmith.wrappers import wrap_openai',
        '# from langsmith.wrappers import wrap_openai'
    )
    result = result.replace(
        'import openai',
        '# import openai\nfrom langchain_aws import ChatBedrockConverse\nfrom langchain_core.messages import HumanMessage, SystemMessage'
    )
    result = result.replace(
        'MODEL_PROVIDER = "openai"',
        '# MODEL_PROVIDER = "openai"\nMODEL_PROVIDER = "bedrock"'
    )
    result = result.replace(
        'MODEL_NAME = "gpt-4o-mini"',
        f'# MODEL_NAME = "gpt-4o-mini"\nMODEL_NAME = "{NOVA2_LITE}"'
    )
    result = result.replace(
        'openai_client = wrap_openai(openai.Client())',
        '# openai_client = wrap_openai(openai.Client())\nbedrock_client = ChatBedrockConverse(model=MODEL_NAME)'
    )
    result = result.replace(
        '    return call_openai(messages)',
        '    # return call_openai(messages)\n    return call_bedrock(messages)'
    )
    result = result.replace(
        '#@traceable\ndef call_openai(\n    messages: List[dict],\n) -> str:\n    return openai_client.chat.completions.create(\n        model=MODEL_NAME,\n        messages=messages,\n    )',
        '# #@traceable\n# def call_openai(\n#     messages: List[dict],\n# ) -> str:\n#     return openai_client.chat.completions.create(\n#         model=MODEL_NAME,\n#         messages=messages,\n#     )\n\ndef call_bedrock(messages: list) -> str:\n    lc_messages = []\n    for m in messages:\n        if m["role"] == "system":\n            lc_messages.append(SystemMessage(content=m["content"]))\n        else:\n            lc_messages.append(HumanMessage(content=m["content"]))\n    return bedrock_client.invoke(lc_messages)'
    )
    result = result.replace(
        'return response.choices[0].message.content',
        '# return response.choices[0].message.content\n    return response.content'
    )
    result = result.replace(
        'langsmith_rag_with_wrap_openai',
        'langsmith_rag_with_bedrock'
    )
    return result

def migrate_alt_wrap_openai_simple_call(text):
    """Simple openai_client.chat.completions.create call."""
    result = text
    result = result.replace(
        'openai_client.chat.completions.create(\n    model=MODEL_NAME,\n    messages=messages,\n    langsmith_extra={"metadata": {"foo": "bar"}},\n)',
        '# openai_client.chat.completions.create(\n#     model=MODEL_NAME,\n#     messages=messages,\n#     langsmith_extra={"metadata": {"foo": "bar"}},\n# )\nbedrock_client.invoke([HumanMessage(content="What color is the sky?")])'
    )
    return result

def migrate_alt_runtree_cell(text):
    """Cell with RunTree + OpenAI."""
    result = text
    result = result.replace(
        'from openai import OpenAI',
        '# from openai import OpenAI\nfrom langchain_aws import ChatBedrockConverse\nfrom langchain_core.messages import HumanMessage, SystemMessage'
    )
    result = result.replace(
        'openai_client = OpenAI()',
        '# openai_client = OpenAI()\nbedrock_client = ChatBedrockConverse(model="' + NOVA2_LITE + '")'
    )
    # Replace the call_openai function in RunTree context
    result = result.replace(
        'openai_response = openai_client.chat.completions.create(\n        model=model,\n        messages=messages,\n        temperature=temperature,\n    )',
        '# openai_response = openai_client.chat.completions.create(\n    #     model=model,\n    #     messages=messages,\n    #     temperature=temperature,\n    # )\n    lc_msgs = []\n    for m in messages:\n        if m["role"] == "system":\n            lc_msgs.append(SystemMessage(content=m["content"]))\n        else:\n            lc_msgs.append(HumanMessage(content=m["content"]))\n    openai_response = bedrock_client.invoke(lc_msgs)'
    )
    result = result.replace(
        'output = response.choices[0].message.content',
        '# output = response.choices[0].message.content\n    output = response.content'
    )
    result = result.replace(
        '    parent_run: RunTree, messages: List[dict], model: str = "gpt-4o-mini", temperature: float = 0.0',
        '    parent_run: RunTree, messages: List[dict], model: str = "' + NOVA2_LITE + '", temperature: float = 0.0'
    )
    return result

def migrate_alt_env_cell(text):
    """Env cell with OPENAI_API_KEY."""
    return text.replace(
        'os.environ["OPENAI_API_KEY"] = ""',
        '# os.environ["OPENAI_API_KEY"] = ""'
    )

process_notebook(path, {
    0: migrate_alt_env_cell,
    2: migrate_alt_langchain_cell,
    3: migrate_alt_context_cell,
    4: migrate_alt_wrap_openai_cell,
    5: migrate_alt_wrap_openai_simple_call,
    7: migrate_alt_env_cell,
    10: migrate_alt_runtree_cell,
})


# --- module_2 notebooks ---
print("\n📁 module_2/")

# dataset_upload.ipynb - solo env var
path = f"{BASE}/module_2/dataset_upload.ipynb"
process_notebook(path, {0: replace_env_cell})

# evaluators.ipynb
path = f"{BASE}/module_2/evaluators.ipynb"
def migrate_evaluators_env(text):
    return text.replace('os.environ["OPENAI_API_KEY"] = ""', '# os.environ["OPENAI_API_KEY"] = ""')

def migrate_evaluators_llm_judge(text):
    result = text
    result = result.replace(
        'from openai import OpenAI',
        '# from openai import OpenAI\nfrom langchain_aws import ChatBedrockConverse'
    )
    result = result.replace(
        'client = OpenAI()',
        '# client = OpenAI()\nbedrock_client = ChatBedrockConverse(model="' + NOVA2_LITE + '")'
    )
    result = result.replace(
        '    completion = client.beta.chat.completions.parse(\n        model="gpt-4o",\n        messages=[\n            {   \n                "role": "system",\n                "content": (\n                    "You are a semantic similarity evaluator. Compare the meanings of two responses to a question, "\n                    "Reference Response and New Response, where the reference is the correct answer, and we are trying to judge if the new response is similar. "\n                    "Provide a score between 1 and 10, where 1 means completely unrelated, and 10 means identical in meaning."\n                ),\n            },\n            {"role": "user", "content": f"Question: {input_question}\\n Reference Response: {reference_response}\\n Run Response: {run_response}"}\n        ],\n        response_format=Similarity_Score,\n    )\n\n    similarity_score = completion.choices[0].message.parsed',
        '    # Comentado: OpenAI structured output\n    # completion = client.beta.chat.completions.parse(\n    #     model="gpt-4o",\n    #     messages=[...],\n    #     response_format=Similarity_Score,\n    # )\n    # similarity_score = completion.choices[0].message.parsed\n    from langchain_core.messages import HumanMessage, SystemMessage\n    structured_llm = bedrock_client.with_structured_output(Similarity_Score)\n    similarity_score = structured_llm.invoke([\n        SystemMessage(content="You are a semantic similarity evaluator. Compare the meanings of two responses to a question, Reference Response and New Response, where the reference is the correct answer, and we are trying to judge if the new response is similar. Provide a score between 1 and 10, where 1 means completely unrelated, and 10 means identical in meaning."),\n        HumanMessage(content=f"Question: {input_question}\\n Reference Response: {reference_response}\\n Run Response: {run_response}")\n    ])'
    )
    return result

def migrate_evaluators_v2(text):
    result = text
    result = result.replace(
        '    completion = client.beta.chat.completions.parse(\n        model="gpt-4o",\n        messages=[\n            {   \n                "role": "system",\n                "content": (\n                    "You are a semantic similarity evaluator. Compare the meanings of two responses to a question, "\n                    "Reference Response and New Response, where the reference is the correct answer, and we are trying to judge if the new response is similar. "\n                    "Provide a score between 1 and 10, where 1 means completely unrelated, and 10 means identical in meaning."\n                ),\n            },\n            {"role": "user", "content": f"Question: {input_question}\\n Reference Response: {reference_response}\\n Run Response: {run_response}"}\n        ],\n        response_format=Similarity_Score,\n    )\n\n    similarity_score = completion.choices[0].message.parsed',
        '    # Comentado: OpenAI structured output\n    # completion = client.beta.chat.completions.parse(...)\n    # similarity_score = completion.choices[0].message.parsed\n    from langchain_core.messages import HumanMessage, SystemMessage\n    structured_llm = bedrock_client.with_structured_output(Similarity_Score)\n    similarity_score = structured_llm.invoke([\n        SystemMessage(content="You are a semantic similarity evaluator. Compare the meanings of two responses to a question, Reference Response and New Response, where the reference is the correct answer, and we are trying to judge if the new response is similar. Provide a score between 1 and 10, where 1 means completely unrelated, and 10 means identical in meaning."),\n        HumanMessage(content=f"Question: {input_question}\\n Reference Response: {reference_response}\\n Run Response: {run_response}")\n    ])'
    )
    return result

process_notebook(path, {
    1: migrate_evaluators_env,
    2: migrate_evaluators_llm_judge,
    3: migrate_evaluators_v2,
})

# experiments.ipynb
path = f"{BASE}/module_2/experiments.ipynb"
def migrate_experiments_app(text):
    result = text
    # Imports
    result = result.replace(
        'from langchain_openai import OpenAIEmbeddings',
        '# from langchain_openai import OpenAIEmbeddings\nfrom langchain_aws import BedrockEmbeddings'
    )
    result = result.replace(
        'from openai import OpenAI',
        '# from openai import OpenAI\nfrom langchain_aws import ChatBedrockConverse\nfrom langchain_core.messages import HumanMessage, SystemMessage'
    )
    result = result.replace(
        'MODEL_NAME = "gpt-4o"',
        f'# MODEL_NAME = "gpt-4o"\nMODEL_NAME = "{NOVA2_LITE}"'
    )
    result = result.replace(
        'MODEL_PROVIDER = "openai"',
        '# MODEL_PROVIDER = "openai"\nMODEL_PROVIDER = "bedrock"'
    )
    result = result.replace(
        'openai_client = OpenAI()',
        '# openai_client = OpenAI()\nbedrock_client = ChatBedrockConverse(model=MODEL_NAME)'
    )
    result = result.replace(
        '    embd = OpenAIEmbeddings()',
        '    # embd = OpenAIEmbeddings()\n    embd = BedrockEmbeddings()'
    )
    result = result.replace(
        '    return call_openai(messages)',
        '    # return call_openai(messages)\n    return call_bedrock(messages)'
    )
    result = result.replace(
        'def call_openai(messages: List[dict]) -> str:\n    return openai_client.chat.completions.create(\n        model=MODEL_NAME,\n        messages=messages,\n    )',
        '# def call_openai(messages: List[dict]) -> str:\n#     return openai_client.chat.completions.create(\n#         model=MODEL_NAME,\n#         messages=messages,\n#     )\n\ndef call_bedrock(messages: List[dict]) -> str:\n    lc_messages = []\n    for m in messages:\n        if m["role"] == "system":\n            lc_messages.append(SystemMessage(content=m["content"]))\n        else:\n            lc_messages.append(HumanMessage(content=m["content"]))\n    return bedrock_client.invoke(lc_messages)'
    )
    result = result.replace(
        'return response.choices[0].message.content',
        '# return response.choices[0].message.content\n    return response.content'
    )
    # Fix docstring references
    result = result.replace('Calls `call_openai`', 'Calls `call_bedrock`')
    result = result.replace('call_openai\n- Returns the chat completion output from OpenAI', 'call_bedrock\n- Returns the chat completion output from Amazon Bedrock')
    return result

def migrate_experiments_prefix_gpt4o(text):
    return text.replace('experiment_prefix="gpt-4o"', 'experiment_prefix="nova-2-lite"')

def migrate_experiments_prefix_gpt35(text):
    return text.replace('experiment_prefix="gpt-3.5-turbo"', 'experiment_prefix="nova-micro"')

process_notebook(path, {
    0: replace_env_cell,
    2: migrate_experiments_app,
    3: migrate_experiments_prefix_gpt4o,
    4: migrate_experiments_prefix_gpt35,
})

# summary_evaluators.ipynb  
path = f"{BASE}/module_2/summary_evaluators.ipynb"
def migrate_summary_eval_env(text):
    return text.replace('os.environ["OPENAI_API_KEY"] = ""', '# os.environ["OPENAI_API_KEY"] = ""')

def migrate_summary_eval_classifier(text):
    result = text
    result = result.replace(
        'from openai import OpenAI\nopenai_client = OpenAI()',
        '# from openai import OpenAI\n# openai_client = OpenAI()\nfrom langchain_aws import ChatBedrockConverse\nbedrock_client = ChatBedrockConverse(model="' + NOVA2_LITE + '")'
    )
    result = result.replace(
        '    completion = openai_client.beta.chat.completions.parse(\n        model="gpt-4o",\n        messages=[\n            {\n                "role": "user",\n                "content": f"This is the statement: {inputs[\'statement\']}"\n            }\n        ],\n        response_format=Toxicity,\n    )\n\n    toxicity_score = completion.choices[0].message.parsed.toxicity',
        '    # Comentado: OpenAI structured output\n    # completion = openai_client.beta.chat.completions.parse(\n    #     model="gpt-4o", ..., response_format=Toxicity,\n    # )\n    # toxicity_score = completion.choices[0].message.parsed.toxicity\n    from langchain_core.messages import HumanMessage\n    structured_llm = bedrock_client.with_structured_output(Toxicity)\n    parsed = structured_llm.invoke([HumanMessage(content=f"This is the statement: {inputs[\'statement\']}")])\n    toxicity_score = parsed.toxicity'
    )
    return result

process_notebook(path, {
    0: migrate_summary_eval_env,
    2: migrate_summary_eval_classifier,
})

# pairwise_experiments.ipynb
path = f"{BASE}/module_2/pairwise_experiments.ipynb"
def migrate_pairwise_env(text):
    return text.replace('os.environ["OPENAI_API_KEY"] = ""', '# os.environ["OPENAI_API_KEY"] = ""')

def migrate_pairwise_evaluator(text):
    result = text
    result = result.replace(
        'from openai import OpenAI\n\nopenai_client = OpenAI()',
        '# from openai import OpenAI\n# openai_client = OpenAI()\nfrom langchain_aws import ChatBedrockConverse\nfrom langchain_core.messages import HumanMessage, SystemMessage\nbedrock_client = ChatBedrockConverse(model="' + NOVA2_LITE + '")'
    )
    # Replace summary_score_evaluator's parse call
    result = result.replace(
        '    completion = openai_client.beta.chat.completions.parse(\n        model="gpt-4o",\n        messages=[\n            {   \n                "role": "system",\n                "content": SUMMARIZATION_SYSTEM_PROMPT,\n            },\n            {\n                "role": "user",\n                "content": SUMMARIZATION_HUMAN_PROMPT.format(\n                    transcript=inputs["transcript"],\n                    summary=outputs.get("output", "N/A"),\n                )}\n        ],\n        response_format=SummarizationScore,\n    )\n\n    summary_score = completion.choices[0].message.parsed.score',
        '    # Comentado: OpenAI structured output\n    # completion = openai_client.beta.chat.completions.parse(model="gpt-4o", ..., response_format=SummarizationScore)\n    # summary_score = completion.choices[0].message.parsed.score\n    structured_llm = bedrock_client.with_structured_output(SummarizationScore)\n    parsed = structured_llm.invoke([\n        SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),\n        HumanMessage(content=SUMMARIZATION_HUMAN_PROMPT.format(transcript=inputs["transcript"], summary=outputs.get("output", "N/A")))\n    ])\n    summary_score = parsed.score'
    )
    return result

def migrate_pairwise_good_summarizer(text):
    result = text
    result = result.replace(
        '    response = openai_client.chat.completions.create(\n        model="gpt-4o",\n        messages=[\n            {\n                "role": "user",\n                "content": f"Concisely summarize this meeting in 3 sentences. Make sure to include all of the important events. Meeting: {inputs[\'transcript\']}"\n            }\n        ],\n    )\n    return response.choices[0].message.content',
        '    # Comentado: OpenAI\n    # response = openai_client.chat.completions.create(model="gpt-4o", ...)\n    # return response.choices[0].message.content\n    response = bedrock_client.invoke([HumanMessage(content=f"Concisely summarize this meeting in 3 sentences. Make sure to include all of the important events. Meeting: {inputs[\'transcript\']}")])\n    return response.content'
    )
    return result

def migrate_pairwise_bad_summarizer(text):
    result = text
    result = result.replace(
        '    response = openai_client.chat.completions.create(\n        model="gpt-4o",\n        messages=[\n            {\n                "role": "user",\n                "content": f"Summarize this in one sentence. {inputs[\'transcript\']}"\n            }\n        ],\n    )\n    return response.choices[0].message.content',
        '    # Comentado: OpenAI\n    # response = openai_client.chat.completions.create(model="gpt-4o", ...)\n    # return response.choices[0].message.content\n    response = bedrock_client.invoke([HumanMessage(content=f"Summarize this in one sentence. {inputs[\'transcript\']}")])\n    return response.content'
    )
    return result

def migrate_pairwise_preference(text):
    result = text
    result = result.replace(
        '    completion = openai_client.beta.chat.completions.parse(\n        model="gpt-4o",\n        messages=[\n            {   \n                "role": "system",\n                "content": JUDGE_SYSTEM_PROMPT,\n            },\n            {\n                "role": "user",\n                "content": JUDGE_HUMAN_PROMPT.format(\n                    transcript=inputs["transcript"],\n                    answer_a=outputs[0].get("output", "N/A"),\n                    answer_b=outputs[1].get("output", "N/A")\n                )}\n        ],\n        response_format=Preference,\n    )\n\n    preference_score = completion.choices[0].message.parsed.preference',
        '    # Comentado: OpenAI structured output\n    # completion = openai_client.beta.chat.completions.parse(model="gpt-4o", ..., response_format=Preference)\n    # preference_score = completion.choices[0].message.parsed.preference\n    from langchain_core.messages import HumanMessage, SystemMessage\n    structured_llm = bedrock_client.with_structured_output(Preference)\n    parsed = structured_llm.invoke([\n        SystemMessage(content=JUDGE_SYSTEM_PROMPT),\n        HumanMessage(content=JUDGE_HUMAN_PROMPT.format(transcript=inputs["transcript"], answer_a=outputs[0].get("output", "N/A"), answer_b=outputs[1].get("output", "N/A")))\n    ])\n    preference_score = parsed.preference'
    )
    return result

process_notebook(path, {
    0: migrate_pairwise_env,
    2: migrate_pairwise_evaluator,
    3: migrate_pairwise_good_summarizer,
    4: migrate_pairwise_bad_summarizer,
    6: migrate_pairwise_preference,
})

# --- module_3 ---
print("\n📁 module_3/")

# prompt_hub.ipynb
path = f"{BASE}/module_3/prompt_hub.ipynb"
def migrate_prompt_hub_env(text):
    return text.replace('os.environ["OPENAI_API_KEY"] = ""', '# os.environ["OPENAI_API_KEY"] = ""')

def migrate_prompt_hub_openai_call(text):
    result = text
    result = result.replace(
        'from openai import OpenAI\nfrom langsmith.client import convert_prompt_to_openai_format\n\nopenai_client = OpenAI()\n\n# NOTE: We can use this utility from LangSmith to convert our hydrated prompt to openai format\nconverted_messages = convert_prompt_to_openai_format(hydrated_prompt)["messages"]\n\nopenai_client.chat.completions.create(\n        model="gpt-4o-mini",\n        messages=converted_messages,\n    )',
        '# from openai import OpenAI\n# from langsmith.client import convert_prompt_to_openai_format\n# openai_client = OpenAI()\n# converted_messages = convert_prompt_to_openai_format(hydrated_prompt)["messages"]\n# openai_client.chat.completions.create(model="gpt-4o-mini", messages=converted_messages)\n\nfrom langchain_aws import ChatBedrockConverse\nbedrock_client = ChatBedrockConverse(model="' + NOVA2_LITE + '")\nbedrock_client.invoke(hydrated_prompt.to_messages())'
    )
    return result

def migrate_prompt_hub_openai_call2(text):
    result = text
    result = result.replace(
        'from openai import OpenAI\nfrom langsmith.client import convert_prompt_to_openai_format\n\nopenai_client = OpenAI()\n\nhydrated_prompt = prompt.invoke({"question": "What is the world like?", "language": "English"})\n# NOTE: We can use this utility from LangSmith to convert our hydrated prompt to openai format\nconverted_messages = convert_prompt_to_openai_format(hydrated_prompt)["messages"]\n\nopenai_client.chat.completions.create(\n        model="gpt-4o-mini",\n        messages=converted_messages,\n    )',
        '# from openai import OpenAI\n# from langsmith.client import convert_prompt_to_openai_format\n# openai_client = OpenAI()\n# converted_messages = convert_prompt_to_openai_format(hydrated_prompt)["messages"]\n# openai_client.chat.completions.create(model="gpt-4o-mini", messages=converted_messages)\n\nfrom langchain_aws import ChatBedrockConverse\nbedrock_client = ChatBedrockConverse(model="' + NOVA2_LITE + '")\nhydrated_prompt = prompt.invoke({"question": "What is the world like?", "language": "English"})\nbedrock_client.invoke(hydrated_prompt.to_messages())'
    )
    return result

def migrate_prompt_hub_chatopenai(text):
    result = text
    result = result.replace(
        'from langchain_openai import ChatOpenAI',
        '# from langchain_openai import ChatOpenAI\nfrom langchain_aws import ChatBedrockConverse'
    )
    result = result.replace(
        'model = ChatOpenAI(model="gpt-4o-mini")',
        '# model = ChatOpenAI(model="gpt-4o-mini")\nmodel = ChatBedrockConverse(model="' + NOVA2_LITE + '")'
    )
    return result

process_notebook(path, {
    0: migrate_prompt_hub_env,
    3: migrate_prompt_hub_openai_call,
    7: migrate_prompt_hub_openai_call2,
    8: migrate_prompt_hub_chatopenai,
})

# prompt_engineering_lifecycle.ipynb
path = f"{BASE}/module_3/prompt_engineering_lifecycle.ipynb"
with open(path, 'r') as f:
    nb = json.load(f)
code_idx = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src_text = ''.join(cell['source'])
        if 'OPENAI_API_KEY' in src_text:
            cell['source'] = [line.replace('os.environ["OPENAI_API_KEY"] = ""', '# os.environ["OPENAI_API_KEY"] = ""') for line in cell['source']]
        if 'MODEL_NAME = "gpt-4o-mini"' in src_text:
            new_source = []
            for line in cell['source']:
                if 'MODEL_NAME = "gpt-4o-mini"' in line:
                    new_source.append('# ' + line)
                    new_source.append(f'MODEL_NAME = "{NOVA2_LITE}"\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
        code_idx += 1
with open(path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"  ✅ prompt_engineering_lifecycle.ipynb")


# --- module_5 ---
print("\n📁 module_5/")

# online_evaluation.ipynb
path = f"{BASE}/module_5/online_evaluation.ipynb"
process_notebook(path, {0: replace_env_cell})

# filtering.ipynb
path = f"{BASE}/module_5/filtering.ipynb"
process_notebook(path, {0: replace_env_cell})

print("\n✅ Migration complete!")
