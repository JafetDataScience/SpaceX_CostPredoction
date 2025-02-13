from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import uvicorn
import nest_asyncio
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#from langchain.chains import RetrievalQA
#import gradio as gr
nest_asyncio.apply()
#os.environ["USER_AGENT"] = "MyFastAPIApp/1.0 (contact@yourdomain.com)"
app = FastAPI()

############# ADDED CORS CONFIG###############
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #allow all origns
    allow_credentials=True,
    allow_methods=["*"], # allow all methods
    allow_headers=["*"] # allow all headers

)
###########################

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

#Generate data base
file_1 = [
    "https://portfolio-showcase-789.framer.ai/",
    "https://portfolio-showcase-789.framer.ai/dataprojects",
    "https://portfolio-showcase-789.framer.ai/SpaceX",
    "https://portfolio-showcase-789.framer.ai/Laptops",
    "https://portfolio-showcase-789.framer.ai/rainoccurance",
    "https://portfolio-showcase-789.framer.ai/universeexpanssion",
    "Newsletter_Economy_1.pdf",
    "FL_solution_ecuation.pdf",
    "resume_QTRO_JILS.pdf"
    ]

## LLM using Hugging Face API
def get_llm(prompt, T=0.5):
    payload = {
        "inputs": prompt,
        "parameters": {
        "max_new_tokens": 256,
        "temperature":T,
        "repetition_penalty":1.2,
        "top_p":0.9 #Controls diversity of generated text
          }
      }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    output = response.json()
    if isinstance(output, list):
        return output[0].get("generated_text", "Error: No response from model")
    return output.get("generated_text", "Error: No response from model")

## Document loader
#file_1 = "FL_solution_ecuation.pdf"
def document_loader(files):
    documents = []
    for file_ in files:
        if file_.startswith("https"):
            loader = WebBaseLoader(file_)#
        else:
            loader = PyPDFLoader(file_)
        documents.extend(loader.load())
    return documents

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

## Vector database
def vector_database(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# QA Chain

memory = ConversationBufferMemory(return_messages=True)
def retriever_qa(query, T=0.5, file=file_1):
    #retriever_obj =
    retrieved_docs = retriever(file).invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    history = memory.load_memory_variables({}).get("history",[])

    # Make sure history is a list and append the conversation as a new item in the list
    prompt = f"History: {''.join(history)}\nContext: {context}\nQuestion: {query}\nAnswer:"
    response = get_llm(prompt, T)
    answer = response.split("Answer:")[-1].strip()  # Extract only the answer
    conversation = f"\nUser: {query}\nBot: {answer}"
    history.append(conversation)
    memory.save_context({"query":query},{"history":history})
    return answer

#Render flask api

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

#@app.head("/") # add HEAD Handler
#async def head_root():
#    return {"message":"HEAD request handled"}

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    user_input = data["question"]
    response = retriever_qa(user_input)
    return {"response": response}
#@app.get("/")
#async def query(request: Request):
#    data = await request.json()
#    user_input = data["question"]
#    response = retriever_qa(user_input)
#    return {"response": response}

if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=7860)
    config = uvicorn.Config(app, port=7860, host="0.0.0.0")
    server = uvicorn.Server(config)
    server.run()
