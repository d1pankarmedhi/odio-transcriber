from core.model.whisper import Whisper


class Transcriber:
    def __init__(self) -> None:
        self.model = Whisper()

    def run(self, audio_file: bytes):
        return self.model.run(audio_file)
