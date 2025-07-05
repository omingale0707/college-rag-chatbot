from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

URI = "/home/zoro/dev/college-rag-chatbot/kirti_college_documents.db"

vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
        auto_id=True
    )

retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k":5}
)


print(retriever.invoke("What all programs are offered for masters in science"))
# print(vector_store.similarity_search("What all programs are offered for masters in science"))