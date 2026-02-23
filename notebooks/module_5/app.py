import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import SKLearnVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain_aws import BedrockEmbeddings
from langsmith import traceable
# from openai import OpenAI
from langchain_aws import ChatBedrockConverse
from typing import List
import nest_asyncio

# MODEL_NAME = "gpt-4o-mini"
# MODEL_PROVIDER = "openai"
MODEL_NAME = "us.amazon.nova-2-lite-v1:0"
MODEL_PROVIDER = "bedrock"
APP_VERSION = 1.0
RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
"""

# openai_client = OpenAI()
bedrock_client = ChatBedrockConverse(model=MODEL_NAME)

def get_vector_db_retriever():
    persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")
    # embd = OpenAIEmbeddings()
    embd = BedrockEmbeddings()

    # If vector store exists, then load it
    if os.path.exists(persist_path):
        vectorstore = SKLearnVectorStore(
            embedding=embd,
            persist_path=persist_path,
            serializer="parquet"
        )
        return vectorstore.as_retriever(lambda_mult=0)

    # Otherwise, index LangSmith documents and create new vector store
    ls_docs_sitemap_loader = SitemapLoader(
        web_path="https://docs.langchain.com/sitemap.xml",
        filter_urls=["https://docs.langchain.com/langsmith/"],
        continue_on_failure=True)
    ls_docs = ls_docs_sitemap_loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(ls_docs)

    # Limit chunks to avoid Bedrock rate limiting (throttling)
    doc_splits = doc_splits[:20]
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embd,
        persist_path=persist_path,
        serializer="parquet"
    )
    vectorstore.persist()
    return vectorstore.as_retriever(lambda_mult=0)

nest_asyncio.apply()
retriever = get_vector_db_retriever()

"""
retrieve_documents
- Returns documents fetched from a vectorstore based on the user's question
"""
@traceable(run_type="chain")
def retrieve_documents(question: str):
    return retriever.invoke(question)

"""
generate_response
- Calls `call_bedrock` to generate a model response after formatting inputs
"""
@traceable(run_type="chain")
def generate_response(question: str, documents):
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context: {formatted_docs} \n\n Question: {question}"
        }
    ]
    return call_bedrock(messages)

"""
call_bedrock
- Returns the chat completion output from Amazon Bedrock
"""
@traceable(
    run_type="llm",
    metadata={
        "ls_provider": MODEL_PROVIDER,
        "ls_model_name": MODEL_NAME
    }
)
def call_bedrock(messages: List[dict]) -> str:
    # Comentado: OpenAI original
    # return openai_client.chat.completions.create(
    #     model=MODEL_NAME,
    #     messages=messages,
    # )
    from langchain_core.messages import HumanMessage, SystemMessage
    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        else:
            lc_messages.append(HumanMessage(content=m["content"]))
    return bedrock_client.invoke(lc_messages)

"""
langsmith_rag
- Calls `retrieve_documents` to fetch documents
- Calls `generate_response` to generate a response based on the fetched documents
- Returns the model response
"""
@traceable(run_type="chain")
def langsmith_rag(question: str):
    documents = retrieve_documents(question)
    response = generate_response(question, documents)
    # Comentado: OpenAI original
    # return response.choices[0].message.content
    return response.content
