from fastapi import FastAPI, File, UploadFile, Request, WebSocket, status, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from utils import load_split_pdf_file, build_history_aware_retriever, build_qa_chain
from dotenv import load_dotenv

load_dotenv()
docs_dir = '../documents'
db_dir = '../db'

templates = Jinja2Templates(directory = "../templates")
app = FastAPI()
app.mount("/static", StaticFiles(directory = "../static"), name = "static")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.6)
# llm = ChatOpenAI(model = "gpt-4o", temperature = 0.6)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@app.get("/", response_class = HTMLResponse)
def return_homepage(request: Request):
    return templates.TemplateResponse(request = request, name = "index.html")

# def create_db_from_file(uploaded_file):
#     docs = load_split_pdf_file(f'{docs_dir}/{uploaded_file.filename}')
#     db = Milvus.from_documents(persist_directory = db_dir,
#                                documents = docs, 
#                                embedding = embedding)
    
@app.post("/")
def send_to_chat():
    # if file.filename.endswith('.pdf'):
    #     contents = file.file.read()
    #     with open(f'{docs_dir}/{file.filename}', 'wb') as f:
    #         f.write(contents)
    #     file.file.close()
    # # background_tasks.add_task(create_db_from_file, file)
    return RedirectResponse(url = "/chatting",
                            status_code = status.HTTP_303_SEE_OTHER
                            )

@app.get("/chatting", response_class = HTMLResponse)
def return_homepage(request: Request):
    return templates.TemplateResponse(request = request, name = "chatting2.html")

@app.websocket("/chatting")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    URI = "/home/zoro/dev/college-rag-chatbot/kirti_college_documents.db"

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
        auto_id=True
    )
    retriever = vector_store.as_retriever()

    history_aware_retriever = build_history_aware_retriever(llm, retriever)
    qa_chain = build_qa_chain(llm)
    history_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        history_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    while True:
        user_input = await websocket.receive_text()

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": "default"}
            },  
        )["answer"]
        await websocket.send_text(response)
