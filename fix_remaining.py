#!/usr/bin/env python3
"""Fix remaining uncommented OpenAI references across all notebooks."""
import json, os

BASE = "/home/juansebas7ian/intro-to-langsmith/notebooks"
NOVA2_LITE = "us.amazon.nova-2-lite-v1:0"

def fix_notebook(path):
    with open(path, 'r') as f:
        nb = json.load(f)
    
    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        new_source = cell['source']
        
        # Skip if already migrated (has ChatBedrockConverse)
        has_bedrock = 'ChatBedrockConverse' in src
        
        # Fix uncommented "from openai import OpenAI"
        needs_fix = False
        for line in src.split('\n'):
            s = line.strip()
            if ('from openai import OpenAI' in s or 'import openai' in s or 
                'from langchain_openai import ChatOpenAI' in s or
                'from langchain_openai import OpenAIEmbeddings' in s) and not s.startswith('#'):
                needs_fix = True
                break
        
        if needs_fix:
            new_lines = []
            added_bedrock_import = False
            for line in cell['source']:
                stripped = line.rstrip('\n').strip()
                
                if 'from openai import OpenAI' in stripped and not stripped.startswith('#'):
                    new_lines.append('# ' + line if not line.startswith('# ') else line)
                    if not has_bedrock and not added_bedrock_import:
                        new_lines.append('from langchain_aws import ChatBedrockConverse\n')
                        new_lines.append('from langchain_core.messages import HumanMessage, SystemMessage\n')
                        added_bedrock_import = True
                    changed = True
                elif 'import openai' in stripped and not stripped.startswith('#') and 'from openai' not in stripped:
                    new_lines.append('# ' + line if not line.startswith('# ') else line)
                    if not has_bedrock and not added_bedrock_import:
                        new_lines.append('from langchain_aws import ChatBedrockConverse\n')
                        added_bedrock_import = True
                    changed = True
                elif 'openai_client = OpenAI()' in stripped and not stripped.startswith('#'):
                    new_lines.append('# ' + line if not line.startswith('# ') else line)
                    if not has_bedrock:
                        new_lines.append(f'bedrock_client = ChatBedrockConverse(model="{NOVA2_LITE}")\n')
                    changed = True
                elif 'client = OpenAI()' in stripped and not stripped.startswith('#') and 'openai' not in stripped.lower().split('=')[0]:
                    new_lines.append('# ' + line if not line.startswith('# ') else line)
                    if not has_bedrock:
                        new_lines.append(f'bedrock_client = ChatBedrockConverse(model="{NOVA2_LITE}")\n')
                    changed = True
                elif 'openai_client.chat.completions.create(' in stripped and not stripped.startswith('#'):
                    new_lines.append('# ' + line if not line.startswith('# ') else line)
                    changed = True
                elif 'openai_client.beta.chat.completions.parse(' in stripped and not stripped.startswith('#'):
                    new_lines.append('# ' + line if not line.startswith('# ') else line)
                    changed = True
                elif 'response.choices[0].message.content' in stripped and not stripped.startswith('#'):
                    new_lines.append('# ' + line if not line.startswith('# ') else line)
                    new_lines.append('    return response.content\n')
                    changed = True
                elif 'MODEL_PROVIDER = "openai"' in stripped and not stripped.startswith('#'):
                    new_lines.append('# ' + line if not line.startswith('# ') else line)
                    new_lines.append('MODEL_PROVIDER = "bedrock"\n')
                    changed = True
                elif ('MODEL_NAME = "gpt-4o-mini"' in stripped or 'MODEL_NAME = "gpt-4o"' in stripped) and not stripped.startswith('#'):
                    new_lines.append('# ' + line if not line.startswith('# ') else line)
                    new_lines.append(f'MODEL_NAME = "{NOVA2_LITE}"\n')
                    changed = True
                elif 'os.environ["OPENAI_API_KEY"]' in stripped and not stripped.startswith('#'):
                    new_lines.append('# ' + line if not line.startswith('# ') else line)
                    changed = True
                else:
                    new_lines.append(line)
            
            cell['source'] = new_lines
            cell['outputs'] = []
            cell['execution_count'] = None
    
    if changed:
        with open(path, 'w') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  🔧 Fixed: {os.path.basename(path)}")
    else:
        print(f"  ✅ OK: {os.path.basename(path)}")

print("🔧 Fixing remaining OpenAI references...\n")

# Process all notebooks
for root, dirs, files in os.walk(BASE):
    for f in sorted(files):
        if f.endswith('.ipynb'):
            fix_notebook(os.path.join(root, f))

print("\n✅ Fix complete!")
