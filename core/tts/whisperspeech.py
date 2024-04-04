from whisperspeech.pipeline import Pipeline


class WhisperSpeechTTS:
    def __init__(
        self, model_id: str = "collabora/whisperspeech:s2a-q4-tiny-en+pl.model"
    ) -> None:
        self.pipe = Pipeline(s2a_ref=model_id)

    def text_to_audio(self, text: str):
        return self.pipe.generate(text)


def load(model_id: str = "collabora/whisperspeech:s2a-q4-tiny-en+pl.model"):
    tts = WhisperSpeechTTS(model_id)
    return tts.text_to_audio("The lion king")


load()
