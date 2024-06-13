from dotenv import load_dotenv

from app.models import ChatbotRequest
from app.models import ChatbotRequestOllama
from app.utils import *

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from starlette.responses import Response
from starlette.middleware.cors import CORSMiddleware

from gradio_client import Client

from ollama import Client as OllamaClient
from ollama import AsyncClient as OllamaAsyncClient

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain import PromptTemplate
#from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

import ast
import time
import asyncio 
import os
import wget


DEBUG = False

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost")
API_PORT = os.getenv("API_PORT", ":8000")

H2OGPT_URL = os.getenv("GRADIO_HOST_URL", "http://localhost")
H2OGPT_PORT = os.getenv("GRADIO_HOST_PORT", ":7860")

OLLAMA_HOST_URL=os.getenv("OLLAMA_HOST_URL","http://localhost")
#OLLAMA_HOST_URL=os.getenv("OLLAMA_HOST_URL","https://24e2-194-199-64-71.ngrok-free.app/")
OLLAMA_HOST_PORT=os.getenv("OLLAMA_HOST_PORT",":11434")

app = FastAPI()

origins = [
    API_URL,
    API_URL+API_PORT,
    H2OGPT_URL,
    H2OGPT_URL+H2OGPT_PORT,
    OLLAMA_HOST_URL,
    OLLAMA_HOST_URL+OLLAMA_HOST_PORT,
    #OLLAMA_HOST_URL,
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Configurate the H2OGPT python client connection
#client = Client(H2OGPT_URL+H2OGPT_PORT)

# Configurate the Ollama python client connection
ollama_client = OllamaClient(host=OLLAMA_HOST_URL+OLLAMA_HOST_PORT)
#ollama_client = OllamaClient(host=OLLAMA_HOST_URL)
ollama_async_client = OllamaAsyncClient(host=OLLAMA_HOST_URL+OLLAMA_HOST_PORT)

async def fake_data_streamer():
    for i in range(10):
        yield b'some fake data\n\n'
        await asyncio.sleep(0.5)

@app.get("/")
def root():
    return {"message": "Fast API in Python"}

@app.get("/test_streaming")
def test_streaming():
    return StreamingResponse(fake_data_streamer(), media_type='text/event-stream')

@app.post("/query_chatbot", status_code=200)
def nochat_api(request: ChatbotRequest):
    api_name = "/submit_nochat_api"
    kwargs = request.__dict__

    res = client.predict(str(dict(kwargs)), api_name=api_name)

    response = ast.literal_eval(res)['response']

    if not response:
        raise HTTPException(status_code=400, detail="Error")

    return response

@app.post("/query_chatbot_plain", status_code=200)
def nochat_api(request: ChatbotRequest):
    api_name = "/submit_nochat_plain_api"
    kwargs = request.__dict__

    res = client.predict(str(dict(kwargs)), api_name=api_name)

    response = ast.literal_eval(res)['response']

    if not response:
        raise HTTPException(status_code=400, detail="Error")

    return response

#We cannot stream on "/submit_nochat_plain_api" endpoint
@app.post("/query_chatbot_stream", status_code=200)
def nochat_api(request: ChatbotRequest):
    api_name = "/submit_nochat_api"
    kwargs = request.__dict__
    job = client.submit(str(dict(kwargs)), api_name=api_name)  
    return StreamingResponse(stream_output(job), media_type='text/event-stream')

#get model_names available
@app.get("/model_names",status_code=200)
def chat_names():
    api_name="/model_names"
    res=ast.literal_eval(client.predict(api_name=api_name))
    models_dict = dict()
    for index, model in enumerate(res):
        models_dict["model_"+str(index)] = model['base_model']
        print(model['base_model'])

    return str(dict(models_dict))

@app.post("/query_specific_model")
def query_specific_model(request: ChatbotRequest):
    request_json = request.__dict__
    if not request_json["visible_models"]:
        raise HTTPException(status_code=404, detail="Missing model to query")

    api_name = "/submit_nochat_api"

    res = client.predict(str(dict(request_json)), api_name=api_name)

    if DEBUG:
        print("Raw client result: %s" % res, flush=True)
    
    res_dict = dict(prompt=request_json['instruction_nochat'], iinput=request_json['iinput_nochat'],
                    response=res)

    return res_dict

# Query Ollama API using the Ollama python client
@app.post("/query_ollama", status_code=200)
def ollama_api(request: ChatbotRequestOllama):

    response = ollama_client.chat(model=request.model, messages=request.messages)

    print(response)

    if not response:
        raise HTTPException(status_code=400, detail="Error")
    
    current_directory = os.getcwd()
    print(current_directory)

    return response

# Query Ollama API using the Ollama python client with stream
@app.post("/query_ollama_stream", status_code=200)
async def ollama_stream(request: ChatbotRequestOllama):
    async def stream_parts():
        async for part in await ollama_async_client.chat(model=request.model, messages=request.messages, stream=True):
            yield part['message']['content']

    return StreamingResponse(stream_parts(), media_type='text/event-stream')

    
    

@app.post("/ollama_rag", status_code=200)
def ollama_rag(request: ChatbotRequestOllama):
    ollama_embedding = OllamaEmbeddings(model="all-minilm",base_url=OLLAMA_HOST_URL)
    db = Chroma(persist_directory="./chroma_db", embedding_function=ollama_embedding, collection_name="test")

    template = """You are an expert programmer that writes simple, concise code and explanations.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = Ollama(model=request.model, base_url=OLLAMA_HOST_URL)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": request.prompt})
    print (result)
    return result

#import a model from Huggingface
@app.post("/ollama_import_HF", status_code=200)
def ollama_import_HF(request:ChatbotRequestOllama):
    download=wget.download(request.url_hf, out='./models/')
    if not download:
        raise HTTPException(status_code=400, detail="Error downloading the model")
    
    name_model=request.url_hf.split("/")[-1]
    
    #create a modelfile from the downloaded model using ollama
    modelfile=f'''
    FROM ~/PyJs_API/models/{name_model}
    '''
    #create a model from the modelfile
    result=ollama_client.create(model=request.name_hf, modelfile=modelfile)

    file_to_remove=f'./models/{name_model}'
    if os.path.isfile(file_to_remove):
        os.remove(file_to_remove)
        
    print(result)
    return result


@app.post("/ollama_preload_model", status_code=200)
def ollama_preload_model(request:ChatbotRequestOllama):
    result=ollama_client.chat(model=request.model)
    print(result)
    return result





"""
@app.get("/alternatives/{question_id}")
def read_alternatives(question_id: int):
    return api.read_alternatives(question_id)


@app.post("/answer", status_code=201)
def create_answer(payload: UserAnswer):
    payload = payload.dict()

    return api.create_answer(payload)


@app.get("/result/{user_id}")
def read_result(user_id: int):
    return api.read_result(user_id)
"""