from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Optional
from model_handler import ModelHandler

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_handler = ModelHandler()

@app.post("/generate")
def infer(
    model_id: str = Form(),
    character_name: str = Form(),
    persona: str = Form(),
    prompt: str = Form(),
    chat_history: Optional[str] = Form(None)
):
    output = _model_handler.generate(model_id, character_name, persona, prompt, chat_history)
    return Response(output)