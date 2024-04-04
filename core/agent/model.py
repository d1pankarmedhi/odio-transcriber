from langchain.chat_models.openai import ChatOpenAI

from core.transcribe.whisper import Whisper


class Model:
    @staticmethod
    def openai_chat_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.01):
        return ChatOpenAI(
            model=model,
            temperature=temperature,
        )

    @staticmethod
    def transcribe_pipeline(model_id: str = "openai/whisper-base"):
        model = Whisper(model_id)
        return model.pipeline()
