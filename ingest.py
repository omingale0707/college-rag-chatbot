from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain import hub
from langchain_core.documents import Document
from zipfile import ZipFile
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)s - %(message)s',
    style='%',
    datefmt='%Y-%m-%d %H:%M',
    level=logging.DEBUG,
)

zip_path = "kirti_college_knowledge_base.zip"
extract_path = "data"

logging.info(f'Starting extracting contents from {zip_path}')
with ZipFile(zip_path, 'r') as zip_ref:
    # print(zip_ref.infolist())
    for zip_info in zip_ref.infolist():
       if zip_info.is_dir():
          continue
       
       zip_info.filename = zip_info.filename.removeprefix(f"{zip_path[:zip_path.find('.')]}/")
       zip_ref.extract(zip_info, extract_path)
    logging.info(f'Completed extracting contents from {zip_path} to folder {extract_path}')

length_function = len

text_splitter = RecursiveCharacterTextSplitter.from_language(
    Language.MARKDOWN,
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=length_function,
)

logging.info(f'Starting chunking of data')
all_splits = []
for dir in os.listdir(extract_path):
  logger.debug(f'folder - {dir}')
  for filename in os.listdir(f"{extract_path}/{dir}"):
    logger.debug(f'filename - {filename}')
    if filename.endswith(".md"):
        loader = TextLoader(os.path.join(extract_path, dir, filename))
        data = loader.load()
        all_splits.extend(text_splitter.split_documents(data))
        logging.info(f'Chunking process finished for {filename}')
logging.info(f'Chunking process finished!')

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

URI = "./kirti_college_documents.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    auto_id=True
)

_ = vector_store.add_documents(documents=all_splits)