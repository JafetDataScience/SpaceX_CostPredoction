from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import os
import requests
import uvicorn
import logging
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

############# CORS CONFIG ###############
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Explicit port configuration for Render
PORT = int(os.environ.get("PORT", 8000))
HOST = "0.0.0.0"

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"  # Smaller model
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

# Document paths
#"https://portfolio-showcase-789.framer.ai/dataprojects", "https://portfolio-showcase-789.framer.ai/universeexpanssion",
file_1 = [
    #"https://portfolio-showcase-789.framer.ai/",
    #"https://portfolio-showcase-789.framer.ai/SpaceX",
    #"https://portfolio-showcase-789.framer.ai/Laptops",
    #"https://portfolio-showcase-789.framer.ai/rainoccurance",
    "Newsletter_Economy_1.pdf",
    "FL_solution_ecuation.pdf",
    "resume_QTRO_JILS.pdf"
]

# Preprocess documents during startup
@app.on_event("startup")
async def initialize_services():
    try:
        logger.info("Starting document preprocessing...")
        splits = document_loader(file_1)
        chunks = text_splitter(splits)
        app.state.vectordb = vector_database(chunks)
        app.state.retriever = app.state.vectordb.as_retriever()
        logger.info("Vector database initialized successfully")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

## LLM with error handling
def get_llm(prompt, T=0.5):
    try:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 256,
                "temperature": T,
                "repetition_penalty": 1.2,
                "top_p": 0.8
            }
        }
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"API Error: {response.text}")
            return f"API Error: {response.text}"
            
        output = response.json()
        if isinstance(output, list):
            return output[0].get("generated_text", "Error: Unexpected response format")
        return output.get("generated_text", "Error: No response from model")
        
    except Exception as e:
        logger.error(f"LLM Exception: {str(e)}")
        return f"Error: {str(e)}"

## Document loader with error handling
def document_loader(files):
    documents = []
    for file_ in files:
        try:
            if file_.startswith("https"):
                logger.info(f"Loading web content: {file_}")
                loader = WebBaseLoader(file_)
            else:
                logger.info(f"Loading PDF: {file_}")
                loader = PyPDFLoader(file_)
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from {file_}")
        except Exception as e:
            logger.error(f"Error loading {file_}: {str(e)}")
    return documents

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=950,
        chunk_overlap=110,
        length_function=len,
    )
    return text_splitter.split_documents(data)

## Vector database
def vector_database(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embedding_model)

# QA Chain with memory
memory = ConversationBufferMemory(input_key="query", memory_key="history")
def retriever_qa(query, T=0.5):
    try:
        if not hasattr(app.state, 'retriever'):
            return "System initialization incomplete"
            
        retrieved_docs = app.state.retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        history = memory.load_memory_variables({}).get("history","")

        prompt = f"History: {history}\nContext: {context}\nQuestion: {query}\nAnswer:"
        response = get_llm(prompt, T)
        answer = response.split("Answer:")[-1].strip()
        
        new_history = f"{history}\nUser: {query}\nBot: {answer}"
        memory.save_context({"query": query}, {"history": new_history})
        
        return answer
        
    except Exception as e:
        logger.error(f"QA Error: {str(e)}")
        return f"Processing error: {str(e)}"

# Endpoints
@app.get("/")
async def home():
    return {"status": "online"}

@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
        user_input = data.get("question", "")
        
        if not user_input:
            return {"error": "No question provided"}
            
        # Use thread pool for CPU-bound tasks
        response = await run_in_threadpool(retriever_qa, user_input, 0.4)
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    config = uvicorn.Config(app, port=PORT, host=HOST)
    server = uvicorn.Server(config)
    server.run()
