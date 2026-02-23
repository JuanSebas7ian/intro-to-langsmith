import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)
from langchain_aws import BedrockEmbeddings

print("Testing BedrockEmbeddings...")
embd = BedrockEmbeddings()  # uses AWS_DEFAULT_REGION etc
try:
    res = embd.embed_query("hello world")
    print(f"Embedding successful, len={len(res)}")
except Exception as e:
    print(f"Error: {e}")
