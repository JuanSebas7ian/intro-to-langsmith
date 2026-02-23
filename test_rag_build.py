import os, time, logging
from dotenv import load_dotenv
load_dotenv(".env", override=True)
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_aws import BedrockEmbeddings
import nest_asyncio

logging.basicConfig(level=logging.INFO)

print("Starting loader...")
ls_docs_sitemap_loader = SitemapLoader(
    web_path="https://docs.langchain.com/sitemap.xml",
    filter_urls=["https://docs.langchain.com/langsmith/"],
    continue_on_failure=True)
ls_docs = ls_docs_sitemap_loader.load()

print(f"Loaded {len(ls_docs)} docs")
t0 = time.time()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(ls_docs)
print(f"Split into {len(doc_splits)} chunks in {time.time()-t0:.2f}s")

print("Building vector store...")
embd = BedrockEmbeddings()

t1 = time.time()
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits[:50], # Just test 50!
    embedding=embd,
    persist_path="test_union.parquet",
    serializer="parquet"
)
print(f"Built vector store for 50 docs in {time.time()-t1:.2f}s")
