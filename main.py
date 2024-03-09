import os

import uvicorn
from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from core.transcribe.whisper import Whisper

app = FastAPI()
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/")
async def home():
    return {"Message": "Server running at 8000"}


@router.post("/transcribe/audio")
async def transcribe(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    print("Loading model..")
    whisper = Whisper()
    pipe = whisper.pipeline()
    print("Model loaded...")

    result = pipe(file_path)
    # print(result['text'])

    # delete the saved audio file
    os.remove(file_path)

    response_data = {
        "status": 200,
        "file_name": file.filename,
        "text": result["text"],
        "text_chunks": result["chunks"],
        "model": "_".join(str(whisper.model_id).split("-")),
    }

    return JSONResponse(status_code=200, content=response_data)


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
