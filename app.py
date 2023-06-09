import os
import textwrap
import openai
from langchain.llms import AzureOpenAI, OpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index.vector_stores import RedisVectorStore
from llama_index import LangchainEmbedding
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext
)
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL) # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from dotenv import load_dotenv
load_dotenv('./.env')

# setup Llama Index to use Azure OpenAI
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get the OpenAI model names ex. "text-embedding-ada-002"
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
text_model = os.getenv("OPENAI_TEXT_MODEL")


print(f"Using models: {embedding_model} and {text_model}")

# get the Azure Deployment name for the model
embedding_model_deployment = os.getenv("AZURE_EMBED_MODEL_DEPLOYMENT_NAME")
text_model_deployment = os.getenv("AZURE_TEXT_MODEL_DEPLOYMENT_NAME")

print(f"Using deployments: {embedding_model_deployment} and {text_model_deployment}")

llm = AzureOpenAI(deployment_name=text_model_deployment, model_kwargs={
    "api_key": openai.api_key,
    "api_base": openai.api_base,
    "api_type": openai.api_type,
    "api_version": openai.api_version,
})
llm_predictor = LLMPredictor(llm=llm)

embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model=embedding_model,
        deployment=embedding_model_deployment,
        openai_api_key= openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1,
)

# load documents
documents = SimpleDirectoryReader('./data').load_data()
print('Document ID:', documents[0].doc_id)

# set number of output tokens
num_output = int(os.getenv("OPENAI_MAX_TOKENS"))
# max LLM token input size
max_input_size = int(os.getenv("CHUNK_SIZE"))
# set maximum chunk overlap
max_chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# define the service we will use to answer questions
# if you executive the Azure OpenAI code above, your Azure Models and creds will be used and the same for OpenAI
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
#    prompt_helper=prompt_helper # uncomment to use prompt_helper.
)

def format_redis_conn_from_env(using_ssl=False):
    start = "rediss://" if using_ssl else "redis://"
    # if using RBAC
    password = os.getenv("REDIS_PASSWORD", None)
    username = os.getenv("REDIS_USERNAME", "default")
    if password != None:
        start += f"{username}:{password}@"

    return start + f"{os.getenv('REDIS_ADDRESS')}:{os.getenv('REDIS_PORT')}"


# make using_ssl=True to use SSL with ACRE
redis_address = format_redis_conn_from_env(using_ssl=False)

print(f"Using Redis address: {redis_address}")
vector_store = RedisVectorStore(
    index_name="ttrpg_docs",
    index_prefix="blog",
    redis_url=redis_address,
    overwrite=True
)

# access the underlying client in the RedisVectorStore implementation to ping the redis instance
vector_store.client.ping()

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = GPTVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context
)

query_engine = index.as_query_engine()
response = query_engine.query("what are the races of dungeons and dragons?")
print("\n", textwrap.fill(str(response), 100))

response = query_engine.query("What are the classes of dungeons and dragons?")
print("\n", textwrap.fill(str(response), 100))
