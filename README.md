# odio

## Getting started

Follow the below steps to setup the server in your local machine.

1. Clone the repository
```bash
git clone https://github.com/wavnet/odio.git
```
2. Create a virtual env and Install the necessary requirements
```bash
# create venv
python -m venv .venv

# activate venv
source .venv/bin/activate # linux
.venv\Scripts\activate # windows

# install dependencies
pip install -r requirements.txt
```

4. Start the application
```bash
python main.py
```

5. Make a **POST** request to `transcribe/audio` route. See below example.
```bash
curl -X 'POST' \
  'http://localhost:8000/transcribe/audio' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@Audio_Kevin_Folta.wav;type=audio/wav'
```
