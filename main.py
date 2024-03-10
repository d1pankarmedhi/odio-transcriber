import io
import os

import uvicorn
from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from core.transcribe import transcribe_audio

app = FastAPI()
router = APIRouter()


@router.get("/")
async def home():
    return {"Message": "Server running at 8000"}


@router.post("/transcribe/audio")
async def transcribe(file: UploadFile = File(...)):
    audio_file = await file.read()
    result = transcribe_audio(audio_file)
    response_data = {
        "status": 200,
        "file_name": file.filename,
        "text": result["text"],
        "text_chunks": result["chunks"],
    }

    return JSONResponse(status_code=200, content=response_data)


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
