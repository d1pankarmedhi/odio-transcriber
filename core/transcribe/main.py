from core.transcribe.whisper import Whisper


def transcribe_audio(file):
    whisper = Whisper()
    pipe = whisper.pipeline()

    result = pipe(file)
    return result
