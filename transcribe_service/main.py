import whisper
from fastapi import FastAPI, UploadFile, File
import os 
import uvicorn


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def home():
    return {"Message": "Server running at 8000"}

@app.post("/transcribe/audio")
async def transcribe(file: UploadFile = File(...)):

    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    print(f"Loading model..")
    whisper_model = whisper.load_model("models/base.pt")
    print(f"Model loaded...")

    result = whisper_model.transcribe(file_path)
    # print(result['text'])

    # delete the saved audio file
    os.remove(file_path)

    return {"Message": file.filename, "Text": result['text']}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
