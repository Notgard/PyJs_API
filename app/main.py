import os

from dotenv import load_dotenv

from app.models import ChatbotRequest
from fastapi import FastAPI, HTTPException
from starlette.responses import Response
from starlette.middleware.cors import CORSMiddleware

from gradio_client import Client
import ast
import time

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost")
API_PORT = os.getenv("API_PORT", ":8000")

H2OGPT_URL = os.getenv("GRADIO_HOST_URL", "http://localhost")
H2OGPT_PORT = os.getenv("GRADIO_HOST_PORT", ":7860")

app = FastAPI()

origins = [
    API_URL,
    API_URL+API_PORT,
    H2OGPT_URL,
    H2OGPT_URL+H2OGPT_PORT,
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
client = Client(H2OGPT_URL+H2OGPT_PORT)

@app.get("/")
def root():
    return {"message": "Fast API in Python"}


@app.post("/query_chatbot", status_code=200)
def nochat_api(request: ChatbotRequest):
    api_name = "/submit_nochat_api"
    kwargs = request.__dict__

    res = client.predict(str(dict(kwargs)), api_name=api_name)

    response = ast.literal_eval(res)['response']

    if not response:
        raise HTTPException(status_code=400, detail="Error")

    return response
    
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